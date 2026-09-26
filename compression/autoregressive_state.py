"""Causal state shared by encoding and decoding; context resets are explicit."""
from numbers import Integral
from .model import Prediction


class AutoregressiveState:
    def __init__(self, owner, length):
        self.owner, self.length = owner, length
        self.count = 0
        self.context = [owner.bos]
        self.cache = self.pending = None

    @property
    def done(self):
        return self.count == self.length

    def next_predictions(self):
        import torch
        if self.pending is not None:
            raise RuntimeError("Accept previous predictions first")
        if self.done:
            return []
        owner = self.owner
        cached = owner.settings.use_kv_cache
        ids = self.context[-1:] if cached else self.context
        kwargs = dict(input_ids=torch.tensor([ids], device=owner.device),
                      use_cache=cached, return_dict=True)
        if cached:
            kwargs['past_key_values'] = self.cache
        with torch.inference_mode():
            output = owner.model(**kwargs)
        if cached:
            self.cache = output.past_key_values
            if self.cache is None:
                raise RuntimeError("Causal model did not return a KV cache")
        self.pending = owner.distribution(output, -1)
        return [Prediction(self.count, self.pending)]

    def accept(self, symbols):
        if self.pending is None or len(symbols) != 1:
            raise ValueError("Expected one pending symbol")
        symbol = symbols[0]
        if not isinstance(symbol, Integral) or not 0 <= symbol < self.pending.vocabulary_size:
            raise ValueError("Symbol outside vocabulary")
        self.count += 1
        # A full window predicts this boundary symbol; it seeds the next window.
        if len(self.context) == self.owner.context_limit:
            self.context = [int(symbol)]
            self.cache = None
        else:
            self.context.append(int(symbol))
        self.pending = None

    def close(self):
        self.cache = self.pending = self.context = self.owner = None
