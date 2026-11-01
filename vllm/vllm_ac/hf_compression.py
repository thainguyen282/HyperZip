"""Hugging Face baseline with identical CDF, framing and arithmetic coder."""
from .cdf import integer_cdf, select_intervals, encoding_intervals
from .codec import Encoder, Decoder, pack, unpack, text_metadata, recover_text


class HFCompressor:
    def __init__(self, model, tokenizer, model_id, revision):
        self.model, self.tokenizer = model, tokenizer
        self.identity = dict(backend='hf', model=model_id, revision=revision,
                             dtype=str(model.dtype))

    def _step(self, current, cache):
        import torch
        inputs = torch.tensor([current], device=self.model.device, dtype=torch.long)
        # Request only the last head row for cached sequential compression.
        output = self.model(input_ids=inputs, past_key_values=cache, use_cache=True,
                            logits_to_keep=1)
        return output.logits[:, -1], output.past_key_values

    def compress(self, prompt, target, mode='cached', reference=False):
        import torch
        if mode not in ('cached', 'prefill'):
            raise ValueError('Unknown HF compression mode')
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError('Empty prompt')
        meta = text_metadata(target, ids, prompt_ids, mode=mode, reference=reference,
                             decode_schedule='padded-prefill-v1' if mode == 'prefill' else 'cached-v1',
                             **self.identity)
        recover_text(self.tokenizer, ids, meta)
        encoder = Encoder()
        with torch.inference_mode():
            if mode == 'prefill' and ids:
                inputs = torch.tensor([prompt_ids + ids[:-1]], device=self.model.device)
                output = self.model(input_ids=inputs, use_cache=False)
                for left in range(0, len(ids), 32):
                    logits = output.logits[0, len(prompt_ids)-1+left:len(prompt_ids)-1+left+32]
                    tokens = torch.tensor(ids[left:left+32], device=logits.device)
                    for _, low, high in encoding_intervals(logits, tokens, reference=reference).cpu().tolist():
                        encoder.write(low, high)
            else:
                cache, current = None, prompt_ids
                for token in ids:
                    logits, cache = self._step(current, cache)
                    _, low, high = encoding_intervals(logits, torch.tensor([token]), reference=reference).cpu().tolist()[0]
                    encoder.write(low, high)
                    current = [token]
        return pack(encoder.finish(), meta)

    def decompress(self, archive):
        import torch
        meta, payload = unpack(archive)
        for key, value in self.identity.items():
            if meta.get(key) != value:
                raise ValueError(f'Archive/model mismatch: {key}')
        decoder = Decoder(payload)
        ids, cache, current = [], None, meta['prompt_token_ids']
        with torch.inference_mode():
            for position in range(meta['token_count']):
                if meta.get('decode_schedule') == 'padded-prefill-v1':
                    prompt = meta['prompt_token_ids']
                    padded = prompt + ids + [0] * (meta['token_count'] - position - 1)
                    inputs = torch.tensor([padded], device=self.model.device)
                    output = self.model(input_ids=inputs, use_cache=False)
                    left = (position // 32) * 32
                    right = min(left + 32, meta['token_count'])
                    offset = len(prompt) - 1
                    logits = output.logits[0, offset + left:offset + right]
                    cdf = integer_cdf(logits, reference=meta['reference'])
                    cdf = cdf[position - left:position - left + 1]
                else:
                    logits, cache = self._step(current, cache)
                    cdf = integer_cdf(logits, reference=meta['reference'])
                token, low, high = select_intervals(cdf, values=[decoder.value()]).cpu().tolist()[0]
                decoder.advance(low, high)
                ids.append(token)
                current = [token]
        return recover_text(self.tokenizer, ids, meta)
