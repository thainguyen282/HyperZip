from vllm import LLM
from vllm_ac.core import VLLMPatch


class TeacherForcedScoring(VLLMPatch[LLM]):
    def generate(
        self,
        prompts,
        sampling_params=None,
        *,
        use_tqdm=True,
        lora_request=None,
        priority=None,
        reconstruction_target=None
    ):
        original = TeacherForcedScoring._original_generate
        if reconstruction_target is None:
            return original(
                self,
                prompts,
                sampling_params,
                use_tqdm=use_tqdm,
                lora_request=lora_request,
                priority=priority,
            )
        if lora_request is not None or priority is not None:
            raise ValueError("Experimental scoring does not support LoRA or priority")
        from vllm_ac.scoring import score

        return score(
            self, original, prompts, sampling_params, reconstruction_target, use_tqdm
        )
