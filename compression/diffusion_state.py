"""Deterministic confidence selection and uncached masked-block refinement."""
from contextlib import nullcontext
import math
from numbers import Integral
from .model import Prediction


def select_confident(predictions, threshold):
    """Select all above threshold plus the best; ties use the lowest position."""
    confidence = {p.position: float(p.distribution.probabilities.max()) for p in predictions}
    best = min(confidence, key=lambda p: (-confidence[p], p))
    return [p for p in sorted(predictions, key=lambda p: p.position)
            if p.position == best or confidence[p.position] > threshold]


def validate_feedback(pending, symbols):
    if pending is None or len(symbols) != len(pending):
        raise ValueError("Expected one symbol per pending prediction")
    for prediction, symbol in zip(pending, symbols):
        if not isinstance(symbol, Integral) or not 0 <= symbol < prediction.distribution.vocabulary_size:
            raise ValueError("Symbol outside vocabulary")


class MaskedBlockState:
    """BOS plus bounded masked symbols, refining small blocks without caches."""

    def __init__(self, owner, length):
        import torch
        if length + 1 > owner.max_length:
            raise ValueError("Block plus BOS exceeds model context length")
        self.owner = owner
        self.current = torch.tensor([[owner.bos] + [owner.mask] * length], device=owner.device)
        self.unresolved = set(range(length))
        self.pending = None

    @property
    def done(self):
        return not self.unresolved

    def next_predictions(self):
        import torch
        if self.pending is not None:
            raise RuntimeError("Accept previous predictions first")
        if self.done:
            return []
        owner = self.owner
        size = owner.cache_settings.small_block_size
        start = ((min(self.unresolved) + 1) // size) * size
        positions = [p for p in sorted(self.unresolved) if start <= p + 1 < start + size]
        with torch.inference_mode():
            output = owner.model(
                input_ids=self.current,
                position_ids=torch.arange(self.current.shape[1], device=owner.device)[None],
                block_size=owner.attention_block_size, use_cache=False,
                use_block_cache=False, return_dict=True,
            )
        predictions = [Prediction(p, owner.distribution(output, p)) for p in positions]
        self.pending = select_confident(predictions, owner.cache_settings.confidence_threshold)
        return self.pending

    def accept(self, symbols):
        validate_feedback(self.pending, symbols)
        for prediction, symbol in zip(self.pending, symbols):
            self.current[0, prediction.position + 1] = int(symbol)
            self.unresolved.remove(prediction.position)
        self.pending = None

    def close(self):
        self.owner = self.current = self.pending = None
        self.unresolved.clear()


class NemotronState:
    """Refine all unresolved positions in a block; cache only finalized tokens."""

    def __init__(self, owner, length):
        self.owner, self.length = owner, length
        self.offset = 0
        self.cache = self.current = self.pending = None
        self.unresolved = set()
        self.attention_modules = [module for module in owner.model.modules()
                                  if hasattr(module, "diffusion_lm")]

    @property
    def done(self):
        return self.offset == self.length

    def _forward(self, ids, *, causal=False):
        import torch
        # Native attention switches between bidirectional refinement and causal
        # prefix commits. Restore every switch even if the forward fails.
        modes = [(module, module.diffusion_lm) for module in self.attention_modules]
        backend = nullcontext()
        if self.owner.settings.fast_inference:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            # Causal cache commits have an explicit offset mask. Keep the math
            # backend there; masked-block refinement supports fused attention.
            flash = (not causal and ids.is_cuda
                     and self.owner.model.dtype in (torch.float16, torch.bfloat16))
            backend = sdpa_kernel(SDPBackend.FLASH_ATTENTION if flash else SDPBackend.MATH)
        previous_length = 0 if self.cache is None else self.cache.get_seq_length()
        try:
            for module, _ in modes:
                module.diffusion_lm = not causal
            with backend, torch.inference_mode():
                output = self.owner.model(
                    input_ids=ids, past_key_values=self.cache, use_cache=causal,
                    use_causal_mask=causal, return_dict=True,
                )
            if causal:
                self.cache = output.past_key_values
                if self.cache is None or self.cache.get_seq_length() != previous_length + ids.shape[1]:
                    raise RuntimeError("Nemotron did not commit exactly the finalized prefix")
            elif self.cache is not None and self.cache.get_seq_length() != previous_length:
                raise RuntimeError("Nemotron modified the prefix cache during refinement")
            return output
        finally:
            for module, value in modes:
                module.diffusion_lm = value

    def next_predictions(self):
        import torch
        if self.pending is not None:
            raise RuntimeError("Accept previous predictions first")
        if self.done:
            return []
        owner = self.owner
        if self.current is None:
            size = min(owner.settings.block_size, self.length - self.offset)
            if owner.settings.use_kv_cache:
                # Reset only at block boundaries. Seed each new context with EOS.
                if self.cache is not None and self.cache.get_seq_length() + size > owner.context_limit:
                    self.cache = None
                if self.cache is None:
                    self._forward(torch.tensor([[owner.eos]], device=owner.device), causal=True)
            self.current = torch.full((1, size), owner.mask, dtype=torch.long, device=owner.device)
            self.unresolved = set(range(size))
        output = self._forward(self.current)
        positions = sorted(self.unresolved)
        if owner.settings.fast_inference:
            # Only a small confidence vector crosses to the CPU before selection.
            logits = output.logits[0, positions]
            confidence = torch.softmax(logits.float(), dim=-1).amax(dim=-1).cpu().tolist()
            if not all(math.isfinite(value) for value in confidence):
                raise ValueError("Expected finite Nemotron logits")
            best = max(range(len(positions)), key=confidence.__getitem__)
            positions = [p for i, p in enumerate(positions)
                         if i == best or confidence[i] > owner.settings.confidence_threshold]
        distributions = owner.distributions(output, positions)
        predictions = [Prediction(self.offset + p, distribution)
                       for p, distribution in zip(positions, distributions)]
        self.pending = (predictions if owner.settings.fast_inference
                        else select_confident(predictions, owner.settings.confidence_threshold))
        return self.pending

    def accept(self, symbols):
        validate_feedback(self.pending, symbols)
        import torch
        positions = [prediction.position - self.offset for prediction in self.pending]
        self.current[0, positions] = torch.tensor(symbols, dtype=torch.long, device=self.owner.device)
        self.unresolved.difference_update(positions)
        self.pending = None
        if not self.unresolved:
            size = self.current.shape[1]
            self.offset += size
            if self.owner.settings.use_kv_cache and not self.done:
                self._forward(self.current, causal=True)
            self.current = None

    def close(self):
        self.owner = self.cache = self.current = self.pending = None
        self.attention_modules.clear()
        self.unresolved.clear()
