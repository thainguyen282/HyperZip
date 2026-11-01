"""Version 2 coding distribution with associative integer accumulation.

FP32 softmax -> 30-bit integer weights -> 24-bit positive-frequency CDF.
Encoding reduces only the selected prefix, avoiding a vocabulary-wide scan.
The decoder scans the SAME integer weights, so reduction ordering is irrelevant.
"""
TOTAL = 1 << 24
WEIGHT_SCALE = 1 << 30


def integer_weights(logits, *, reference=False):
    import torch
    vocabulary = logits.shape[-1]
    if not 1 < vocabulary < TOTAL:
        raise ValueError('Vocabulary must fit the 24-bit frequency budget')
    values = logits.float()
    probabilities = values.log_softmax(-1).exp() if reference else values.softmax(-1)
    # Power-of-two scaling is exact; truncation floors nonnegative probabilities.
    return (probabilities * WEIGHT_SCALE).to(torch.int64)


def integer_cdf(logits, *, reference=False):
    import torch
    weights = integer_weights(logits, reference=reference)
    vocabulary = weights.shape[-1]
    cumulative = weights.cumsum(-1)
    upper = torch.div(cumulative * (TOTAL - vocabulary), cumulative[..., -1:], rounding_mode='floor')
    upper += torch.arange(1, vocabulary + 1, device=logits.device)
    return torch.cat((torch.zeros_like(upper[..., :1]), upper), dim=-1)


def encoding_intervals(logits, tokens, *, reference=False):
    import torch
    weights = integer_weights(logits, reference=reference)
    vocabulary = weights.shape[-1]
    tokens = tokens.to(device=logits.device, dtype=torch.int64).reshape(-1, 1)
    indices = torch.arange(vocabulary, device=logits.device)
    lower_mass = torch.where(indices < tokens, weights, 0).sum(-1, keepdim=True)
    target_mass = weights.gather(-1, tokens)
    mass = weights.sum(-1, keepdim=True)
    budget = TOTAL - vocabulary
    low = tokens + torch.div(lower_mass * budget, mass, rounding_mode='floor')
    high = tokens + 1 + torch.div((lower_mass + target_mass) * budget, mass, rounding_mode='floor')
    return torch.cat((tokens, low, high), dim=-1)


def select_intervals(cdf, tokens=None, values=None):
    import torch
    if values is not None:
        values = torch.as_tensor(values, device=cdf.device, dtype=torch.int64).reshape(-1, 1)
        tokens = torch.searchsorted(cdf.contiguous(), values.contiguous(), right=True).squeeze(-1) - 1
    tokens = tokens.to(device=cdf.device, dtype=torch.int64).reshape(-1, 1)
    low = cdf.gather(-1, tokens)
    high = cdf.gather(-1, tokens + 1)
    return torch.cat((tokens, low, high), dim=-1)
