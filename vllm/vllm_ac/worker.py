"""Opt-in vLLM 0.11 V1 worker hook; ordinary prompt scoring delegates unchanged.

The private request flag reserves three LogprobsTensors slots as an internal
transport for (token, low, high), NOT log probabilities. Integers <= 2**24 are
exact in float32. Only the compression API consumes these slots.
"""
KEY = "vllm_ac_interval_v2"


def install():
    import vllm
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if vllm.__version__ != "0.11.0":
        raise RuntimeError("Arithmetic worker requires vLLM 0.11.0")
    original = GPUModelRunner._get_prompt_logprobs_dict
    if getattr(original, "_ac_worker", False):
        return

    def intervals(self, hidden_states, num_scheduled_tokens):
        import torch
        from vllm.v1.outputs import LogprobsTensors
        from .cdf import integer_cdf, select_intervals, encoding_intervals
        pending = self.input_batch.num_prompt_logprobs
        selected = {rid: count for rid, count in pending.items()
                    if (self.requests[rid].sampling_params.extra_args or {}).get(KEY)}
        if not selected:
            return original(self, hidden_states, num_scheduled_tokens)
        # Hide private requests from the original handler, including mixed batches.
        for rid in selected:
            del pending[rid]
        try:
            result = original(self, hidden_states, num_scheduled_tokens)
        finally:
            pending.update(selected)
        in_progress = self.input_batch.in_progress_prompt_logprobs_cpu
        for rid, count in selected.items():
            if count != 2:
                raise RuntimeError("Invalid arithmetic transport width")
            request = self.requests[rid]
            options = request.sampling_params.extra_args[KEY]
            ids = request.prompt_token_ids
            start = request.num_computed_tokens
            rows = min(num_scheduled_tokens[rid], len(ids) - start - 1)
            buffer = in_progress.get(rid)
            if buffer is None:
                buffer = LogprobsTensors.empty_cpu(len(ids) - 1, 3)
                buffer.logprob_token_ids[:] = torch.tensor([0, 1, 2], dtype=torch.int32)
                buffer.logprobs.zero_()
                buffer.selected_token_ranks.fill_(1)
                in_progress[rid] = buffer
            # Do not apply the output head to context-only positions. Limit peak
            # vocabulary memory independently of the scheduler's prefill chunk.
            first = max(start, options['target_start'] - 1)
            stop = start + max(0, rows)
            offset = self.query_start_loc.np[self.input_batch.req_id_to_index[rid]].item()
            for left in range(first, stop, 32):
                right = min(left + 32, stop)
                position = options.get('decode_position')
                selected_row = None if position is None else options['target_start'] - 1 + position
                if selected_row is not None and not left <= selected_row < right:
                    continue
                states = hidden_states[offset + left - start:offset + right - start]
                logits = self.model.compute_logits(states)
                targets = torch.tensor(ids[left + 1:right + 1], device=logits.device)
                value = options.get('decode_value')
                if value is None:
                    packed = encoding_intervals(logits, targets)
                else:
                    cdf = integer_cdf(logits)
                    if selected_row is not None:
                        # Preserve the encoder's head and softmax row counts,
                        # then select only the row currently being decoded.
                        row = selected_row - left
                        cdf = cdf[row:row + 1]
                    packed = select_intervals(cdf, values=[value])
                destination = (slice(left, right) if selected_row is None else
                               slice(selected_row, selected_row + 1))
                buffer.logprobs[destination].copy_(packed.float(), non_blocking=True)
            if num_scheduled_tokens[rid] > len(ids) - start - 1:
                result[rid] = buffer
                del pending[rid]
                del in_progress[rid]
        if result:
            self._sync_device()
        return result

    intervals._ac_worker = True
    GPUModelRunner._get_prompt_logprobs_dict = intervals
