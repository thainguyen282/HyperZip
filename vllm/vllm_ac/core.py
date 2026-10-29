"""Small targeted patch framework following the vLLM plugin guide."""

import inspect


class VLLMPatch:
    @classmethod
    def __class_getitem__(cls, target):
        return type(f"VLLMPatch[{target.__name__}]", (cls,), {"_target": target})

    @classmethod
    def apply(cls):
        import vllm

        if vllm.__version__ != "0.11.0":
            raise RuntimeError("TeacherForcedScoring requires vLLM 0.11.0")
        original = cls._target.generate
        if getattr(original, "_teacher_forced_patch", False):
            return
        expected = [
            "self",
            "prompts",
            "sampling_params",
            "use_tqdm",
            "lora_request",
            "priority",
        ]
        if list(inspect.signature(original).parameters) != expected:
            raise RuntimeError("Unsupported LLM.generate signature")
        cls._original_generate = staticmethod(original)
        cls.generate._teacher_forced_patch = True
        cls._target.generate = cls.generate


class PatchManager:
    def __init__(self):
        self.available_patches = {}
        self.applied_patches = set()

    def register(self, name, patch):
        if name in self.available_patches and self.available_patches[name] is not patch:
            raise RuntimeError(f"Conflicting patch: {name}")
        self.available_patches[name] = patch

    def apply_patch(self, name):
        if name not in self.applied_patches:
            self.available_patches[name].apply()
            self.applied_patches.add(name)


manager = PatchManager()
