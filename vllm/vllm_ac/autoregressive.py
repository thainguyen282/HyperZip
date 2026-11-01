"""Native autoregressive arithmetic coding through vLLM's sampling hook."""
CODING_KEY = 'vllm_ac_arithmetic_v1'


def install():
    import inspect
    import vllm
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if vllm.__version__ != '0.11.0':
        raise RuntimeError('Autoregressive compression requires vLLM 0.11.0')
    original = GPUModelRunner._sample
    if getattr(original, '_ac_arithmetic', False):
        return
    if list(inspect.signature(original).parameters) != ['self', 'logits', 'spec_decode_metadata']:
        raise RuntimeError('Unsupported GPUModelRunner._sample signature')

    def arithmetic(self, logits, spec_decode_metadata):
        requests = [(row, rid, (self.requests[rid].sampling_params.extra_args or {}).get(CODING_KEY))
                     for row, rid in enumerate(self.input_batch.req_ids)]
        if not any(options is not None for _, _, options in requests):
            return original(self, logits, spec_decode_metadata)
        if any(options is None for _, _, options in requests):
            raise RuntimeError('Arithmetic calls require an isolated native batch')
        if spec_decode_metadata is not None or self.use_async_scheduling:
            raise RuntimeError('Arithmetic coding requires synchronous non-speculative scheduling')
        try:
            return _coding_sample(self, logits, requests)
        except BaseException:
            self._ac_states = {}
            self._ac_results = {}
            raise

    arithmetic._ac_arithmetic = True
    GPUModelRunner._sample = arithmetic


def _coding_sample(runner, logits, requests):
    """One native sampling step; only integer intervals cross to the CPU."""
    import torch
    from vllm.v1.outputs import SamplerOutput
    from .cdf import encoding_intervals, integer_cdf, select_intervals
    from .codec import Encoder, Decoder
    if not hasattr(runner, '_ac_states'):
        runner._ac_states, runner._ac_results = {}, {}
    discarded = set(runner.discard_request_indices.np[:runner.num_discarded_requests].tolist())
    active = []
    new_requests = 0
    for row, rid, options in requests:
        if row in discarded:
            continue
        position = len(runner.requests[rid].output_token_ids)
        state = runner._ac_states.get(rid)
        if state is None:
            if position:
                raise RuntimeError('Missing arithmetic request state')
            new_requests += 1
            coder = Decoder(options['payload']) if options['decode'] else Encoder()
            state = dict(coder=coder, position=0, options=options)
            runner._ac_states[rid] = state
        if position != state['position'] or position >= options['count']:
            raise RuntimeError('Arithmetic token position mismatch')
        active.append((row, rid, state))
    runner._ac_admissions = getattr(runner, '_ac_admissions', 0) + new_requests
    if new_requests and new_requests < len(active):
        runner._ac_refills = getattr(runner, '_ac_refills', 0) + new_requests
    sampled = torch.zeros((logits.shape[0], 1), dtype=torch.int32, device=logits.device)
    if active:
        modes = {state['options']['decode'] for _, _, state in active}
        if len(modes) != 1:
            raise RuntimeError('Cannot mix arithmetic encode and decode requests')
        rows = torch.tensor([row for row, _, _ in active], device=logits.device)
        values = logits if len(active) == logits.shape[0] else logits.index_select(0, rows)
        if modes == {True}:
            packed = select_intervals(integer_cdf(values),
                values=[state['coder'].value() for _, _, state in active])
        else:
            tokens = torch.tensor([state['options']['targets'][state['position']]
                                   for _, _, state in active], device=logits.device)
            packed = encoding_intervals(values, tokens)
        sampled[rows, 0] = packed[:, 0].to(torch.int32)
        for (_, rid, state), (token, low, high) in zip(active, packed.cpu().tolist()):
            options = state['options']
            if options['decode']:
                state['coder'].advance(low, high)
            else:
                state['coder'].write(low, high)
            state['position'] += 1
            if state['position'] == options['count']:
                key = (options['call'], options['index'])
                runner._ac_results[key] = None if options['decode'] else state['coder'].finish()
                del runner._ac_states[rid]
        stats = getattr(runner, '_ac_stats', {})
        stats[len(active)] = stats.get(len(active), 0) + 1
        runner._ac_stats = stats
    return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)


def coding_results(worker, call):
    """One RPC after generate, never one RPC per token. Also cleans failed calls."""
    runner = worker.model_runner
    results = getattr(runner, '_ac_results', {})
    selected = {index: value for (owner, index), value in results.items() if owner == call}
    runner._ac_results = {key: value for key, value in results.items() if key[0] != call}
    runner._ac_states = {rid: state for rid, state in getattr(runner, '_ac_states', {}).items()
                         if state['options']['call'] != call}
    return selected
