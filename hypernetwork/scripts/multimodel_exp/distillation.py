"""Token-level, teacher-detached KL for matching diffusion predictions."""
import math
import torch
import torch.nn.functional as F


def masked_distillation_kl(student_logits, teacher_logits, temperature=2.0):
    """Mean KL(teacher || student), over selected positions; teacher is detached."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Distillation temperature must be finite and positive')
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError('Student and teacher must have matching [positions, vocabulary] logits')
    if student_logits.shape[0] == 0:
        raise ValueError('Distillation requires at least one position')
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).mean() * temperature ** 2
