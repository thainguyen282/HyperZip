"""Arithmetic coding through the patched LLM.generate pipeline."""
import copy
import os

from .codec import Encoder, Decoder, pack, unpack, text_metadata, recover_text
from .worker import KEY


def _identity(llm):
    import torch
    import vllm
    config = llm.llm_engine.model_config
    return dict(backend='vllm', model=config.model, revision=config.revision,
                tokenizer=config.tokenizer, tokenizer_revision=config.tokenizer_revision,
                dtype=str(config.dtype), torch_version=torch.__version__, vllm_version=vllm.__version__)


def _require_prefill_config(llm):
    if not llm.llm_engine.model_config.enforce_eager:
        raise ValueError('Prefill compression requires enforce_eager=True')
    if llm.llm_engine.cache_config.enable_prefix_caching:
        raise ValueError('Prefill compression requires enable_prefix_caching=False')


def _params(llm, supplied, start, value=None, decode_position=None):
    from vllm import SamplingParams
    params = copy.deepcopy(supplied if supplied is not None else llm.get_default_sampling_params())
    if not isinstance(params, SamplingParams) or params.n != 1 or params.truncate_prompt_tokens is not None:
        raise ValueError('Compression requires n=1 and no prompt truncation')
    params.prompt_logprobs = 2
    params.logprobs = None
    params.max_tokens = 1
    # The required continuation is discarded; avoid stochastic sampling work.
    params.temperature = 0.0
    params.top_p = 1.0
    params.top_k = -1
    params.min_p = 0.0
    params.min_tokens = 0
    params.stop = None
    params.stop_token_ids = []
    params.ignore_eos = False
    params.detokenize = False
    params.extra_args = dict(params.extra_args or {})
    params.extra_args[KEY] = dict(target_start=start, decode_value=value, decode_position=decode_position)
    return params


def _intervals(llm, original, ids, start, supplied=None, value=None, decode_position=None):
    if len(ids) + 1 > llm.llm_engine.model_config.max_model_len:
        raise ValueError('Input plus continuation exceeds context window')
    output = original(llm, [{'prompt_token_ids': ids}],
                      _params(llm, supplied, start, value, decode_position), use_tqdm=False)[0]
    if output.prompt_token_ids != ids or output.prompt_logprobs is None:
        raise RuntimeError('Invalid arithmetic worker output')
    result = []
    rows = (output.prompt_logprobs[start:] if decode_position is None else
            [output.prompt_logprobs[start + decode_position]])
    for row in rows:
        try:
            token, low, high = [row[i].logprob for i in range(3)]
            if any(int(x) != x for x in (token, low, high)) or not 0 <= low < high <= 1 << 24:
                raise ValueError()
            result.append((int(token), int(low), int(high)))
        except (KeyError, TypeError, ValueError, OverflowError):
            raise RuntimeError('Arithmetic worker hook is inactive or returned invalid intervals') from None
    if len(result) != (len(ids) - start if decode_position is None else 1):
        raise RuntimeError('Missing arithmetic intervals')
    return result


def compress(llm, original, prompts, supplied, target, mode):
    if os.getenv("VLLM_AC_ENABLE") != "1":
        raise RuntimeError("Set VLLM_AC_ENABLE=1 before starting Python and the engine")
    if mode not in ('prefix', 'prefill'):
        raise ValueError("compression_mode must be 'prefix' or 'prefill'")
    if mode == 'prefill':
        _require_prefill_config(llm)
    if isinstance(prompts, (list, tuple)) and len(prompts) == 1:
        prompts = prompts[0]
    if not isinstance(prompts, str) or not isinstance(target, str):
        raise TypeError('Compression requires one text prompt and target')
    tok = llm.get_tokenizer()
    prompt = list(tok.encode(prompts, add_special_tokens=True))
    ids = list(tok.encode(target, add_special_tokens=False))
    if not prompt:
        raise ValueError('Prompt must contain at least one token')
    meta = text_metadata(target, ids, prompt, mode=mode,
                         decode_schedule='padded-prefill-v1' if mode == 'prefill' else 'prefix-v1',
                         **_identity(llm))
    # Reject tokenizers that normalize text rather than silently losing bytes.
    recover_text(tok, ids, meta)
    _params(llm, supplied, len(prompt))
    if len(prompt) + len(ids) + bool(ids) > llm.llm_engine.model_config.max_model_len:
        raise ValueError('Input plus continuation exceeds context window')
    encoder = Encoder()
    if mode == 'prefill' and ids:
        rows = _intervals(llm, original, prompt + ids, len(prompt), supplied)
        for expected, (token, low, high) in zip(ids, rows):
            if expected != token:
                raise RuntimeError('Target alignment mismatch')
            encoder.write(low, high)
    elif ids:
        # Encoder and decoder use identical prefix lengths. The final input
        # token cannot affect the preceding hidden state under causal attention.
        prefix = prompt.copy()
        for token in ids:
            # The worker takes the actual token to select its interval, but this
            # position is causally AFTER the hidden state used for that interval.
            row = _intervals(llm, original, prefix + [token], len(prefix), supplied)[0]
            if row[0] != token:
                raise RuntimeError('Target alignment mismatch')
            encoder.write(row[1], row[2])
            prefix.append(token)
    return pack(encoder.finish(), meta)


def decompress(llm, archive, *, compression_mode=None, prompt=None):
    from .patches.teacher_forced import TeacherForcedScoring
    from .bundle import MAGIC, unpack_bundle
    if compression_mode == 'compact':
        return _decompress_compact(llm, TeacherForcedScoring._original_generate, archive, prompt)[0]
    if compression_mode is not None or prompt is not None:
        raise ValueError('Explicit compression_mode and prompt are only supported for compact decoding')
    if archive[:4] == MAGIC:
        records = unpack_bundle(archive)
        if records and unpack(records[0])[0].get('mode') == 'autoregressive':
            return _decompress_autoregressive(llm, TeacherForcedScoring._original_generate, records)
        return ''.join(decompress(llm, record) for record in records)
    meta, payload = unpack(archive)
    if meta.get('mode') == 'autoregressive':
        raise ValueError('Autoregressive chunks must be decoded together from their bundle')
    for key, value in _identity(llm).items():
        if meta.get(key) != value:
            raise ValueError(f'Archive/model mismatch: {key}')
    if meta.get('decode_schedule') == 'padded-prefill-v1':
        _require_prefill_config(llm)
    prefix = list(meta['prompt_token_ids'])
    if not prefix or len(prefix) + meta['token_count'] + bool(meta['token_count']) > llm.llm_engine.model_config.max_model_len:
        raise ValueError('Invalid archive context length')
    decoder = Decoder(payload)
    ids = []
    for position in range(meta['token_count']):
        if meta.get('decode_schedule') == 'padded-prefill-v1':
            # Fixed full input length preserves the transformer GEMM/attention
            # shapes. Future padding lies beyond the causal prediction position.
            padded = prefix + [0] * (meta['token_count'] - position)
            token, low, high = _intervals(
                llm, TeacherForcedScoring._original_generate, padded,
                len(meta['prompt_token_ids']), value=decoder.value(),
                decode_position=position)[0]
        else:
            token, low, high = _intervals(llm, TeacherForcedScoring._original_generate,
                                         prefix + [0], len(prefix), value=decoder.value())[0]
        decoder.advance(low, high)
        prefix.append(token)
        ids.append(token)
    return recover_text(llm.get_tokenizer(), ids, meta)


def _ar_settings(llm):
    """The supported reproducibility contract is one fixed, offline engine."""
    from vllm import envs
    config = llm.llm_engine.vllm_config
    if llm.llm_engine.has_unfinished_requests():
        raise ValueError('Arithmetic generation requires an idle engine')
    if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        raise ValueError('Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before engine initialization')
    if config.parallel_config.world_size != 1 or config.speculative_config is not None:
        raise ValueError('Arithmetic generation requires one GPU and no speculative decoding')
    if config.cache_config.enable_prefix_caching or config.scheduler_config.async_scheduling:
        raise ValueError('Disable prefix caching and async scheduling for arithmetic generation')
    import sys
    import torch
    from vllm.config import CUDAGraphMode
    if (config.compilation_config.level != 0
            or config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or config.model_config.enforce_eager):
        raise ValueError("Autoregressive coding requires compilation_config="
                         "{'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY'} and enforce_eager=False")
    if config.cache_config.num_gpu_blocks_override is None:
        raise ValueError('Autoregressive coding requires a fixed num_gpu_blocks_override')
    if (not torch.are_deterministic_algorithms_enabled()
            or os.getenv('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
            or os.getenv('PYTHONHASHSEED') != '0' or sys.flags.hash_randomization
            or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Set PYTHONHASHSEED=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python; '
                         'call torch.use_deterministic_algorithms(True) and disable matmul TF32 before engine initialization')
    settings = dict(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.getenv('CUBLAS_WORKSPACE_CONFIG'),
        python_hash_seed=os.getenv('PYTHONHASHSEED'),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        attention_backend=os.getenv('VLLM_ATTENTION_BACKEND'))
    for name, fields in dict(
        model_config=['max_model_len', 'enforce_eager', 'seed'],
        scheduler_config=['max_num_seqs', 'max_num_batched_tokens', 'enable_chunked_prefill'],
        cache_config=['block_size', 'cache_dtype', 'gpu_memory_utilization',
                      'num_gpu_blocks', 'num_gpu_blocks_override'],
        compilation_config=['level', 'cudagraph_mode', 'cudagraph_capture_sizes']).items():
        obj = getattr(config, name)
        settings[name] = {field: getattr(obj, field) if isinstance(getattr(obj, field),
            (str, int, float, bool, list, type(None))) else str(getattr(obj, field)) for field in fields}
    return settings


def _ar_params(llm, supplied, options):
    from vllm import SamplingParams
    from .autoregressive import CODING_KEY
    # Construct clean parameters: score requests, constraints and penalties must
    # never alter the coding distribution or end a chunk early.
    if supplied is not None and (not isinstance(supplied, SamplingParams)
            or supplied.n != 1 or supplied.truncate_prompt_tokens is not None):
        raise ValueError('Compression requires n=1 and no prompt truncation')
    return SamplingParams(temperature=0, max_tokens=options['count'],
        ignore_eos=True, detokenize=False, logprobs=None, prompt_logprobs=None,
        extra_args={CODING_KEY: options})


def _ar_generate(llm, original, records, targets=None, supplied=None, use_tqdm=False):
    import uuid
    from .autoregressive import coding_results
    call = uuid.uuid4().hex
    jobs, params, active = [], [], []
    for index, (meta, payload) in enumerate(records):
        count, prompt = meta['token_count'], meta['prompt_token_ids']
        if not prompt or len(prompt) + count > llm.llm_engine.model_config.max_model_len:
            raise ValueError('Nonempty prompt and context-window-compatible chunk required')
        if not count:
            continue
        options = dict(call=call, index=index, count=count, decode=targets is None)
        if targets is None:
            options['payload'] = payload
        else:
            options['targets'] = targets[index]
        jobs.append({'prompt_token_ids': prompt})
        params.append(_ar_params(llm, supplied, options))
        active.append(index)
    retrieved = False
    try:
        outputs = original(llm, jobs, params, use_tqdm=use_tqdm) if jobs else []
        completed = llm.collective_rpc(coding_results, args=(call,))[0] if jobs else {}
        retrieved = True
        if len(outputs) != len(active) or set(completed) != set(active):
            raise RuntimeError('Incomplete arithmetic generation or inactive hook')
        result = {}
        for index, output in zip(active, outputs):
            meta = records[index][0]
            if len(output.outputs) != 1 or output.prompt_token_ids != meta['prompt_token_ids']:
                raise RuntimeError('Arithmetic request alignment mismatch')
            ids = list(output.outputs[0].token_ids)
            if len(ids) != meta['token_count'] or (targets is not None and ids != targets[index]):
                raise RuntimeError('Arithmetic token alignment mismatch')
            result[index] = (ids, completed[index])
        return result
    finally:
        if jobs and not retrieved:
            try:
                engine = llm.llm_engine
                engine.abort_request(list(engine.output_processor.request_states))
            finally:
                llm.collective_rpc(coding_results, args=(call,))


def compress_autoregressive(llm, original, prompts, supplied, chunks, output_path, use_tqdm):
    from pathlib import Path
    from .bundle import pack_bundle
    if output_path is None:
        raise ValueError('Autoregressive compression requires output_path')
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    settings = _ar_settings(llm)
    chunks = [chunks] if isinstance(chunks, str) else chunks
    if not isinstance(chunks, (list, tuple)) or not chunks or not all(isinstance(c, str) for c in chunks):
        raise TypeError('Expected a nonempty list of text chunks')
    if not isinstance(prompts, str):
        raise TypeError('Autoregressive compression requires one shared text prompt')
    tokenizer = llm.get_tokenizer()
    prompt = list(tokenizer.encode(prompts, add_special_tokens=True))
    records, targets = [], []
    for index, text in enumerate(chunks):
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        meta = text_metadata(text, ids, prompt, mode='autoregressive',
            decode_schedule='native-ar-v1', chunk_index=index, chunk_count=len(chunks),
            execution_settings=settings, **_identity(llm))
        recover_text(tokenizer, ids, meta)
        records.append((meta, b''))
        targets.append(ids)
    generated = _ar_generate(llm, original, records, targets, supplied, use_tqdm)
    archives = [pack(generated[index][1] if index in generated else Encoder().finish(), meta)
                for index, (meta, _) in enumerate(records)]
    data = pack_bundle(archives)
    with path.open('xb') as stream:
        try:
            stream.write(data)
        except BaseException:
            path.unlink()
            raise
    return path


def _decompress_autoregressive(llm, original, archives):
    settings = _ar_settings(llm)
    records = [unpack(archive) for archive in archives]
    for index, (meta, _) in enumerate(records):
        expected = dict(_identity(llm), mode='autoregressive', decode_schedule='native-ar-v1',
            execution_settings=settings, chunk_index=index, chunk_count=len(records))
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f'Archive/model mismatch: {key}')
    generated = _ar_generate(llm, original, records)
    return ''.join(recover_text(llm.get_tokenizer(), generated[index][0] if index in generated else [], meta)
                   for index, (meta, _) in enumerate(records))


def _compact_prompt(llm, prompt):
    if not isinstance(prompt, str):
        raise TypeError('Compact compression requires the external text prompt')
    ids = list(llm.get_tokenizer().encode(prompt, add_special_tokens=True))
    if not ids or len(ids) > llm.llm_engine.model_config.max_model_len:
        raise ValueError('Nonempty context-window-compatible prompt required')
    return ids


def compress_compact(llm, original, prompt, supplied, source, output_path, use_tqdm):
    """Encode exact token slices without detokenizing individual chunks."""
    from pathlib import Path
    from .bundle import COMPACT_CHUNK_TOKENS, pack_compact
    if output_path is None:
        raise ValueError('Compact compression requires output_path')
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    if not isinstance(source, str):
        raise TypeError('Compact compression requires one complete source string')
    _ar_settings(llm)
    prompt_ids = _compact_prompt(llm, prompt)
    tokenizer = llm.get_tokenizer()
    ids = list(tokenizer.encode(source, add_special_tokens=False))
    restored = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if restored.encode('utf-8') != source.encode('utf-8'):
        raise ValueError('Tokenizer does not preserve the exact source bytes')
    targets = [ids[start:start + COMPACT_CHUNK_TOKENS]
               for start in range(0, len(ids), COMPACT_CHUNK_TOKENS)]
    records = [(dict(token_count=len(chunk), prompt_token_ids=prompt_ids), b'') for chunk in targets]
    generated = _ar_generate(llm, original, records, targets, supplied, use_tqdm)
    data = pack_compact([generated[index][1] for index in range(len(records))],
                        len(targets[-1]) if targets else 0)
    with path.open('xb') as stream:
        try:
            stream.write(data)
        except BaseException:
            path.unlink()
            raise
    return path


def _decompress_compact(llm, original, archive, prompt):
    """Return text and actual recovered IDs; IDs support external validation."""
    from .bundle import COMPACT_CHUNK_TOKENS, unpack_compact
    final_count, payloads = unpack_compact(archive)
    _ar_settings(llm)
    prompt_ids = _compact_prompt(llm, prompt)
    records = [(dict(token_count=final_count if index == len(payloads) - 1 else COMPACT_CHUNK_TOKENS,
                     prompt_token_ids=prompt_ids), payload)
               for index, payload in enumerate(payloads)]
    generated = _ar_generate(llm, original, records)
    ids = [token for index in range(len(records)) for token in generated[index][0]]
    text = llm.get_tokenizer().decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text, ids
