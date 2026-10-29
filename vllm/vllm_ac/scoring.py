"""Score observed target tokens using native causal prompt log probabilities."""

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


@dataclass(frozen=True)
class TokenScore:
    position: int
    token_id: int
    token_text: str
    log_probability: float
    probability: float


@dataclass
class ScoringResult:
    prompt: str
    target: str
    prompt_token_ids: list[int]
    target_token_ids: list[int]
    tokens: list[TokenScore]
    metadata: dict
    total_negative_log_likelihood: float

    def save_json(self, path):
        # Never silently overwrite an earlier experiment.
        payload = json.dumps(asdict(self), ensure_ascii=True, indent=2, allow_nan=False)
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")


def score(llm, original, prompts, sampling_params, target, use_tqdm):
    import vllm
    from vllm import SamplingParams

    if isinstance(prompts, str):
        prompt = prompts
    elif (
        isinstance(prompts, (list, tuple))
        and len(prompts) == 1
        and isinstance(prompts[0], str)
    ):
        prompt = prompts[0]
    else:
        raise ValueError("Provide exactly one text prompt")
    if not isinstance(target, str):
        raise TypeError("reconstruction_target must be a string")
    params = (
        sampling_params
        if sampling_params is not None
        else llm.get_default_sampling_params()
    )
    if not isinstance(params, SamplingParams) or params.n != 1:
        raise ValueError("Provide one SamplingParams with n=1")
    if params.truncate_prompt_tokens is not None:
        raise ValueError("Prompt truncation is not supported")
    tokenizer = llm.get_tokenizer()
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("The prompt must contain at least one token")
    config = llm.llm_engine.model_config
    combined = prompt_ids + target_ids
    if len(combined) + bool(target_ids) > config.max_model_len:
        raise ValueError(
            "Input plus the required continuation exceeds the context window"
        )
    metadata = {
        "model": config.model,
        "model_revision": config.revision,
        "tokenizer": config.tokenizer,
        "tokenizer_revision": config.tokenizer_revision,
        "vllm_version": vllm.__version__,
        "boundary": "separately_tokenized",
        "probabilities": "native_prompt_logprobs",
    }
    records = []
    if target_ids:
        params = copy.deepcopy(params)
        params.prompt_logprobs = 1
        params.logprobs = None
        params.max_tokens = 1
        params.min_tokens = 0
        params.stop = None
        params.stop_token_ids = []
        params.ignore_eos = False
        outputs = original(
            llm, [{"prompt_token_ids": combined}], params, use_tqdm=use_tqdm
        )
        if len(outputs) != 1 or outputs[0].prompt_token_ids != combined:
            raise RuntimeError("Native output token alignment changed")
        rows = outputs[0].prompt_logprobs
        if rows is None or len(rows) != len(combined):
            raise RuntimeError("Incomplete prompt log probabilities")
        for i, token_id in enumerate(target_ids):
            row = rows[len(prompt_ids) + i]
            if row is None or token_id not in row:
                raise RuntimeError(f"Missing ground-truth score at target position {i}")
            value = float(row[token_id].logprob)
            if not math.isfinite(value) or value > 0:
                raise RuntimeError(f"Invalid ground-truth log probability: {value}")
            records.append(
                TokenScore(
                    i, token_id, tokenizer.decode([token_id]), value, math.exp(value)
                )
            )
    return ScoringResult(
        prompt,
        target,
        prompt_ids,
        target_ids,
        records,
        metadata,
        -math.fsum(record.log_probability for record in records),
    )
