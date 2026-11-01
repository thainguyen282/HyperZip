from vllm import LLM
from vllm_ac.core import VLLMPatch


class TeacherForcedScoring(VLLMPatch[LLM]):
    """Historical plugin name retained for installed configurations; targets compress."""
    def generate(
        self,
        prompts,
        sampling_params=None,
        *,
        use_tqdm=True,
        lora_request=None,
        priority=None,
        reconstruction_target=None,
        arithmetic_encode=False,
        compression_mode="autoregressive",
        output_path=None,
    ):
        """Generate normally, or compress target chunks into output_path.

        arithmetic_encode is accepted for older callers; targets always compress.
        """
        original = TeacherForcedScoring._original_generate
        if reconstruction_target is None:
            if arithmetic_encode or output_path is not None:
                raise ValueError("Compression requires reconstruction_target")
            return original(self, prompts, sampling_params, use_tqdm=use_tqdm,
                            lora_request=lora_request, priority=priority)
        if lora_request is not None or priority is not None:
            raise ValueError("Compression does not support LoRA or priority overrides")
        if compression_mode == "compact":
            from vllm_ac.compression import compress_compact
            return compress_compact(self, original, prompts, sampling_params,
                                    reconstruction_target, output_path, use_tqdm)
        if compression_mode == "autoregressive":
            from vllm_ac.compression import compress_autoregressive
            return compress_autoregressive(self, original, prompts, sampling_params,
                                          reconstruction_target, output_path, use_tqdm)
        if output_path is not None:
            raise ValueError("output_path requires autoregressive or compact compression")
        from vllm_ac.compression import compress
        return compress(self, original, prompts, sampling_params,
                        reconstruction_target, compression_mode)
