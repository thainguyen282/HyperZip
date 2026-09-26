"""Fast-dLLM masked refinement with finalized-prefix KV and optional DualCache."""
from numbers import Integral

from .model import Prediction
from .diffusion_state import select_confident, validate_feedback


def cache_length(cache):
    return 0 if cache is None else int(cache.get_seq_length())


class FastDLLMState:
    """Refine masked blocks, then commit only finalized blocks to the prefix."""

    def __init__(self, owner, length):
        if not isinstance(length, Integral) or length < 0:
            raise ValueError("Record length must be a nonnegative integer")
        self.owner = owner
        self.length = int(length)
        self.count = 0
        self.block_size = owner.attention_block_size
        self.small_block_size = owner.cache_settings.small_block_size
        self.dual_cache = owner.cache_settings.use_dual_cache
        self.context_limit = owner.cache_context_tokens
        eos = owner.tokenizer.eos_token_id
        if eos is None:
            eos = getattr(owner.model.config, "eos_token_id", None)
        if eos is None:
            raise ValueError("Cached Fast-dLLM requires a fixed EOS token")
        self.eos = int(eos)
        self.prefix_cache = self.block_cache = self.pending = None
        self.small_start = None
        self.current = None
        self.unresolved = set()
        if self.length:
            self._new_block(self.eos)

    @property
    def done(self):
        return self.count == self.length

    def _new_block(self, head):
        import torch

        # The first slot is EOS initially, then a symbol predicted by the previous
        # finalized block. Padding is known from length, never from source text.
        unknown = min(self.length - self.count, self.block_size - 1)
        ids = [head] + [self.owner.mask] * unknown + [self.eos] * (self.block_size - 1 - unknown)
        self.current = torch.tensor([ids], dtype=torch.long, device=self.owner.device)
        self.offset = self.count
        self.unresolved = set(range(1, unknown + 1))
        self.block_cache = None
        self.small_start = None

    def _forward(self, ids, *, commit=False, block_cache=None, replace_position=None):
        import torch

        previous_length = cache_length(self.prefix_cache)
        with torch.inference_mode():
            output = self.owner.model(
                input_ids=ids, use_cache=True, past_key_values=self.prefix_cache,
                update_past_key_values=commit, block_size=self.block_size,
                use_block_cache=self.dual_cache and not commit,
                block_past_key_values=block_cache, replace_position=replace_position,
                return_dict=True,
            )
        prefix = output.past_key_values
        if commit:
            if prefix is None or cache_length(prefix) != previous_length + self.block_size:
                raise RuntimeError("Fast-dLLM did not commit exactly one finalized KV block")
            self.prefix_cache = prefix
        elif cache_length(prefix) != previous_length:
            raise RuntimeError("Fast-dLLM changed the read-only prefix cache during refinement")
        if self.dual_cache and not commit:
            self.block_cache = output.block_past_key_values
            if self.block_cache is None or cache_length(self.block_cache) != self.block_size:
                raise RuntimeError("Fast-dLLM did not return a complete intra-block KV cache")
        return output

    def next_predictions(self):
        if self.pending is not None:
            raise RuntimeError("Accept previous predictions first")
        if self.done:
            return []
        if not self.unresolved:
            output = self._forward(self.current, commit=True)
            self.block_cache = None
            self.kind = "head"
            self.pending = [Prediction(self.count, self.owner.distribution(output, -1))]
        else:
            first = min(self.unresolved)
            start = (first // self.small_block_size) * self.small_block_size
            end = min(start + self.small_block_size, self.block_size)
            if start != self.small_start:
                self.block_cache = None
                self.small_start = start
            # Partial logits cannot predict the small block's first token: its
            # predecessor lies outside this span. Refresh full logits until it resolves.
            full = not self.dual_cache or self.block_cache is None or start in self.unresolved
            if full:
                output = self._forward(self.current)
            else:
                output = self._forward(self.current[:, start:end],
                                       block_cache=self.block_cache, replace_position=start)
            threshold = self.owner.cache_settings.confidence_threshold
            positions = [first] if threshold is None else [p for p in sorted(self.unresolved) if start <= p < end]
            predictions = [Prediction(self.offset + p - 1,
                                      self.owner.distribution(output, p - 1 if full else p - start - 1))
                           for p in positions]
            self.kind = "block"
            self.pending = predictions if threshold is None else select_confident(predictions, threshold)
        return self.pending

    def accept(self, symbols):
        validate_feedback(self.pending, symbols)
        self.count += len(symbols)
        if self.kind == "head":
            if cache_length(self.prefix_cache) + self.block_size > self.context_limit:
                self.prefix_cache = None
            self._new_block(int(symbols[0]))
        else:
            for prediction, symbol in zip(self.pending, symbols):
                position = prediction.position - self.offset + 1
                self.current[0, position] = int(symbol)
                self.unresolved.remove(position)
        self.pending = None

    def close(self):
        self.prefix_cache = self.block_cache = self.current = self.pending = self.owner = None
        self.unresolved.clear()
