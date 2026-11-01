"""Optional plugin. Importing this package does not import vLLM."""

import os


def register():
    selected = {
        name.strip() for name in os.getenv("VLLM_CUSTOM_PATCHES", "").split(",")
    }
    if "TeacherForcedScoring" not in selected:
        return
    from .core import manager
    from .patches.teacher_forced import TeacherForcedScoring

    manager.register("TeacherForcedScoring", TeacherForcedScoring)
    manager.apply_patch("TeacherForcedScoring")

    from .autoregressive import install as install_compression

    install_compression()

    if os.getenv("VLLM_AC_ENABLE") == "1":
        from .worker import install

        install()
