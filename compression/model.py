"""Text models and their shared prediction → coding → feedback loop."""
from contextlib import closing
from dataclasses import dataclass
import io
from numbers import Integral

from config import DiffusionConfig, FAST_DLLM_MASK_ID
from tqdm import tqdm

from .arithmeticcoding import ArithmeticDecoder, ArithmeticEncoder, BitInputStream, BitOutputStream
from .probability import ProbabilityDistribution


@dataclass(frozen=True)
class Prediction:
    position: int  # Position within this session, independent of coding order.  
    distribution: ProbabilityDistribution


class Model:
    """Own arithmetic coding; prediction hooks see only resolved feedback."""

    supported_modalities = frozenset({"text"})
    schedule = ""

    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model.eval(), tokenizer, device
        self.last_run_stats = {}
        self.bos = getattr(model.config, "bos_token_id", None)
        if self.bos is None:
            self.bos = tokenizer.bos_token_id
        if self.bos is None:
            raise ValueError("Checkpoint must define a fixed BOS token")
        self.max_length = int(model.config.max_position_embeddings)

    def encode_bytes(self, payload):
        symbols = self.tokenizer.encode(payload.decode("utf-8"), add_special_tokens=False)
        if self.decode_symbols(symbols) != payload:
            raise ValueError("Tokenizer does not preserve the original bytes")
        return symbols

    def decode_symbols(self, symbols):
        return self.tokenizer.decode(
            symbols, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        ).encode("utf-8")

    @staticmethod
    def distribution(output, position):
        logits = output.logits if hasattr(output, "logits") else output["logits"]
        return ProbabilityDistribution.from_logits(logits[0, position].detach().double().cpu().numpy())

    @staticmethod
    def distributions(output, positions):
        """Copy a prediction group once; retain the original FP64 quantization."""
        import torch
        logits = output.logits if hasattr(output, "logits") else output["logits"]
        rows = logits[0, positions].detach()
        if rows.dtype == torch.bfloat16:
            rows = rows.float()  # NumPy cannot represent BF16; FP32 preserves every value.
        rows = rows.cpu().numpy()
        return [ProbabilityDistribution.from_logits(row) for row in rows]

    def compress(self, symbols, block_size=128, precision=24,
                 progress=False, description="Encoding"):
        with io.BytesIO() as buffer:
            bits = BitOutputStream(buffer)
            coder = ArithmeticEncoder(32, bits)
            self._code(len(symbols), coder, block_size, precision, symbols, progress, description)
            coder.finish()
            while bits.numbitsfilled:
                bits.write(0)
            return buffer.getvalue()

    def decompress(self, payload, count, block_size=128, precision=24,
                   progress=False, description="Decoding"):
        with io.BytesIO(payload) as buffer:
            coder = ArithmeticDecoder(32, BitInputStream(buffer))
            return self._code(count, coder, block_size, precision, None, progress, description)

    def _code(self, count, coder, block_size, precision, source, progress, description):
        if not isinstance(count, int) or count < 0 or not isinstance(block_size, int) or block_size < 1:
            raise ValueError("Invalid symbol count or block size")
        if not isinstance(precision, int) or not 1 <= precision <= 30:
            raise ValueError("Frequency precision must be between 1 and 30")
        symbols, vocabulary_size = [None] * count, None
        with closing(self.create_state(count, {"block_size": block_size})) as state:
            groups = resolved_count = largest_group = refinement_passes = 0
            with tqdm(total=count, desc=description, unit="symbol", disable=not progress) as bar:
                while not state.done:
                    # Materialize the entire group before any source lookup or feedback.
                    predictions = list(self.next_predictions(state))
                    if self.is_refinement_group(state):
                        refinement_passes += 1
                    positions = [p.position for p in predictions]
                    if not positions:
                        raise ValueError("Session made no progress before completion")
                    if (any(not isinstance(p, Integral) or not 0 <= p < count for p in positions)
                            or len(set(positions)) != len(positions)
                            or any(symbols[p] is not None for p in positions)):
                        raise ValueError("Session returned duplicate, resolved, or invalid positions")
                    resolved = []
                    for prediction in predictions:
                        distribution = prediction.distribution
                        if not isinstance(distribution, ProbabilityDistribution):
                            raise TypeError("Models must return ProbabilityDistribution")
                        if vocabulary_size is not None and vocabulary_size != distribution.vocabulary_size:
                            raise ValueError("Vocabulary changed within a record")
                        vocabulary_size = distribution.vocabulary_size
                        cdf = distribution.cumulative_frequencies(precision)
                        if source is None:
                            symbol = coder.read(cdf, vocabulary_size)
                        else:
                            symbol = source[prediction.position]
                            if not isinstance(symbol, Integral) or not 0 <= symbol < vocabulary_size:
                                raise ValueError("Symbol outside model vocabulary")
                            symbol = int(symbol)
                            coder.write(cdf, symbol)
                        symbols[prediction.position] = symbol
                        resolved.append(symbol)
                    self.accept(state, resolved)
                    group_size = len(resolved)
                    groups += 1
                    resolved_count += group_size
                    largest_group = max(largest_group, group_size)
                    if progress:
                        bar.set_postfix(
                            groups=groups, group=group_size,
                            avg_group=f"{resolved_count / groups:.2f}", max_group=largest_group,
                            refresh=False,
                        )
                    bar.update(group_size)
            if any(symbol is None for symbol in symbols):
                raise ValueError("Session completed with unresolved positions")
            self.last_run_stats = {"prediction_groups": groups,
                                   "mean_tokens_per_group": resolved_count / groups if groups else None}
            if isinstance(self, (FastDLLMModel, NemotronModel)):
                self.last_run_stats['refinement_passes'] = refinement_passes
            return symbols

    def is_refinement_group(self, state):
        return False

    def create_state(self, symbol_count, settings):
        raise NotImplementedError("Implement create_state for this model")

    def next_predictions(self, state):
        return state.next_predictions()

    def accept(self, state, resolved_symbols):
        state.accept(resolved_symbols)

    def _left_to_right_block(self, length):
        if length + 1 > self.max_length:
            raise ValueError("Block plus BOS exceeds model context length")
        return LeftToRightSession(length, self.predict_next)


class BlockState:
    """Keep legacy coding-block resets while exposing record-relative positions."""

    def __init__(self, length, block_size, session_factory):
        self.length, self.block_size = length, block_size
        self.session_factory = session_factory
        self.offset = 0
        self.session = None

    @property
    def done(self):
        return self.offset == self.length

    def next_predictions(self):
        if self.session is None:
            self.session = self.session_factory(min(self.block_size, self.length - self.offset))
        if self.session.done:
            raise ValueError("Session completed with unresolved positions")
        return [Prediction(self.offset + p.position, p.distribution)
                for p in self.session.next_predictions()]

    def accept(self, symbols):
        self.session.accept(symbols)
        if self.session.done:
            self.session.close()
            self.session = None
            self.offset += min(self.block_size, self.length - self.offset)

    def close(self):
        if self.session is not None:
            self.session.close()
        self.session = self.session_factory = None


class LeftToRightSession:
    """Minimal schedule that advances only after symbol feedback."""

    def __init__(self, length, predict):
        if length < 0:
            raise ValueError("Negative block length")
        self.length = length
        self.predict = predict
        self.prefix = []
        self.pending = None

    @property
    def done(self):
        return len(self.prefix) == self.length

    def next_predictions(self):
        if self.pending is not None:
            raise RuntimeError("Accept the previous predictions before requesting more")
        if self.done:
            return []
        self.pending = self.predict(tuple(self.prefix), self.length)
        return [Prediction(len(self.prefix), self.pending)]

    def accept(self, symbols):
        if self.pending is None or len(symbols) != 1:
            raise ValueError("Expected one symbol for the pending prediction")
        symbol = symbols[0]
        if not isinstance(symbol, Integral) or not 0 <= symbol < self.pending.vocabulary_size:
            raise ValueError("Symbol outside vocabulary")
        self.prefix.append(int(symbol))
        self.pending = None

    def close(self):
        # The predictor is normally a bound method holding the loaded model.
        # Release it as well as token state on success and on coding failures.
        self.predict = self.pending = None
        self.prefix.clear()


class AutoregressiveModel(Model):
    """Hugging Face causal-LM inference with shared encoder/decoder context windows."""

    schedule = "causal_bos_left_to_right_v1"

    def __init__(self, model, tokenizer, device, settings=None):
        super().__init__(model, tokenizer, device)
        self.settings = settings
        if settings is not None:
            self.context_limit = settings.context_tokens or self.max_length
            if self.context_limit > self.max_length:
                raise ValueError("Causal context exceeds model context length")
            self.schedule = ("causal_bos_bounded_kv_v2" if settings.use_kv_cache
                             else "causal_bos_bounded_recompute_v2")

    def create_state(self, symbol_count, settings):
        if self.settings is None:
            return BlockState(symbol_count, settings["block_size"], self._left_to_right_block)
        from .autoregressive_state import AutoregressiveState
        return AutoregressiveState(self, symbol_count)

    def predict_next(self, prefix, length):
        import torch

        ids = torch.tensor([[self.bos, *prefix]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=ids, use_cache=False, return_dict=True)
        return self.distribution(output, -1)


# Compatibility name; causal inference is shared by all autoregressive checkpoints.
QwenModel = AutoregressiveModel


class DiffusionModel(Model):
    """Masked inference, with bounded left-to-right blocks as the legacy default."""

    def create_state(self, symbol_count, settings):
        return BlockState(symbol_count, settings["block_size"], self._left_to_right_block)

    def masked_input(self, prefix, length):
        import torch

        ids = [self.bos, *prefix, *([self.mask] * (length - len(prefix)))]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(ids), device=self.device)[None]
        return x, positions


class OmniModel(DiffusionModel):
    schedule = "omni_bos_shifted_left_to_right_v1"

    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        self.mask = int(model.config.mask_token_id)

    def predict_next(self, prefix, length):
        import torch

        x, positions = self.masked_input(prefix, length)
        with torch.inference_mode():
            output = self.model.forward_dream(
                input_ids=x, position_ids=positions,
                use_cache=False, return_dict=True,
            )
        # Omni predicts the unknown position from its predecessor's raw logits.
        return self.distribution(output, len(prefix))


class FastDLLMModel(DiffusionModel):
    """Masked Fast-dLLM prediction, with the checkpoint's one-token logit shift."""  

    schedule = "fast_dllm_bos_shifted_left_to_right_v1"

    def is_refinement_group(self, state):
        if self.cache_settings.use_kv_cache:
            return state.kind == 'block'
        return self.cache_settings.confidence_threshold is not None

    def __init__(self, model, tokenizer, device, settings=None):
        super().__init__(model, tokenizer, device)
        self.mask = FAST_DLLM_MASK_ID
        self.cache_settings = settings if settings is not None else DiffusionConfig()
        settings = self.cache_settings
        self.attention_block_size = settings.block_size
        self.cache_context_tokens = settings.cache_context_tokens or self.max_length
        if settings.use_kv_cache:
            if not settings.block_size <= self.cache_context_tokens <= self.max_length:
                raise ValueError("KV context must hold a diffusion block and fit the model context")
            cache = "dual" if settings.use_dual_cache else "prefix"
            refinement = "left_to_right_v1" if settings.confidence_threshold is None else "confidence_v2"
            self.schedule = f"fast_dllm_eos_{cache}_kv_{refinement}"
        elif settings.confidence_threshold is not None:
            self.schedule = "fast_dllm_bos_masked_confidence_v2"

    def create_state(self, symbol_count, settings):
        if self.cache_settings.use_kv_cache:
            from .fast_dllm_session import FastDLLMState
            return FastDLLMState(self, symbol_count)
        if self.cache_settings.confidence_threshold is not None:
            from .diffusion_state import MaskedBlockState
            return BlockState(symbol_count, settings["block_size"],
                              lambda length: MaskedBlockState(self, length))
        return super().create_state(symbol_count, settings)

    def predict_next(self, prefix, length):
        import torch

        x, positions = self.masked_input(prefix, length)
        with torch.inference_mode():
            output = self.model(
                input_ids=x, position_ids=positions,
                block_size=self.attention_block_size, use_cache=False,
                use_block_cache=False, return_dict=True,
            )
        # Unknown position len(prefix)+1 uses its predecessor's raw logits.
        return self.distribution(output, len(prefix))


class NemotronModel(DiffusionModel):
    """Same-position masked predictions and optional finalized causal prefix."""

    def is_refinement_group(self, state):
        return True

    def __init__(self, model, tokenizer, device, settings):
        super().__init__(model, tokenizer, device)
        if model.config.model_type != "nemotron_labs_diffusion" or model.config.dlm_paradigm != "bidirectional":
            raise ValueError("Expected a bidirectional Nemotron-Labs-Diffusion checkpoint")
        if settings.use_dual_cache:
            raise ValueError("Nemotron does not support Fast-dLLM DualCache")
        if settings.confidence_threshold is None:
            raise ValueError("Nemotron requires a confidence threshold")
        self.settings = settings
        self.mask = int(model.config.mask_token_id)
        self.eos = tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Nemotron requires a fixed EOS token")
        self.context_limit = settings.cache_context_tokens or self.max_length
        if not settings.block_size <= self.context_limit <= self.max_length:
            raise ValueError("Nemotron blocks and context must fit the checkpoint context")
        if settings.use_kv_cache and settings.block_size == self.context_limit:
            raise ValueError("Nemotron KV context must also hold the initial EOS token")
        self.schedule = ("nemotron_eos_bounded_prefix_kv_confidence_v1" if settings.use_kv_cache
                         else "nemotron_masked_block_confidence_v1")
        if settings.fast_inference:
            self.schedule = self.schedule.replace("_confidence_v1", "_fast_confidence_v2")

    def create_state(self, symbol_count, settings):
        from .diffusion_state import NemotronState
        return NemotronState(self, symbol_count)
