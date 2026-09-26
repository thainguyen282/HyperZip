"""The only prediction representation consumed by the arithmetic codec."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class ProbabilityDistribution:
    """One next-symbol distribution; index i always represents symbol ID i.

    Models own output extraction and position selection. Vocabulary sizes
    may differ between models; this class standardizes their interface,
    not their vocabularies. Arrays are copied and made read-only.
    """

    probabilities: np.ndarray

    def __post_init__(self):
        p = np.array(self.probabilities, dtype=np.float64, copy=True)
        if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Expected a finite, nonnegative one-dimensional distribution")
        largest = p.max()
        if largest <= 0:
            raise ValueError("Probability mass must be positive")
        p /= largest  # Avoid overflow when normalizing unnormalized weights.
        p /= p.sum()
        p.setflags(write=False)
        object.__setattr__(self, "probabilities", p)

    @classmethod
    def from_logits(cls, logits):
        values = np.array(logits, dtype=np.float64, copy=True)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("Expected finite one-dimensional logits")
        with np.errstate(over="ignore", under="ignore"):
            return cls(np.exp(values - values.max()))

    @property
    def vocabulary_size(self):
        return len(self.probabilities)

    def cumulative_frequencies(self, precision=24):
        """Positive integer frequencies bounded by the 32-bit coder's budget.

        Reserve one count per symbol, then floor the remaining probability mass.
        Both sides must use the same precision and this exact quantization rule.
        """
        if not isinstance(precision, int) or not 1 <= precision <= 30:
            raise ValueError("Frequency precision must be an integer from 1 to 30")
        total = 1 << precision
        if self.vocabulary_size > total:
            raise ValueError("Vocabulary exceeds the frequency budget")
        counts = np.floor(self.probabilities * (total - self.vocabulary_size)).astype(np.int64) + 1
        return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
