"""Frozen document encoder; last-token pooling and L2 normalization (GTE)."""
import torch
from torch.nn import functional as F

DEFAULT_ENCODER = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'


class DocumentEncoder:
    def __init__(self, model_path=DEFAULT_ENCODER, *, revision=None, max_length=32000,
                 device='cpu', dtype='float32', trust_remote_code=False, local_files_only=False):
        from transformers import AutoModel, AutoTokenizer
        if max_length < 1:
            raise ValueError('Encoder max_length must be positive')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision,
            trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_path, revision=revision,
            trust_remote_code=trust_remote_code, local_files_only=local_files_only,
            torch_dtype=getattr(torch, dtype)).to(device).eval().requires_grad_(False)
        self.device, self.max_length = device, max_length
        self.metadata = dict(model_path=model_path,
            revision=getattr(self.model.config, '_commit_hash', None) or revision,
            max_length=max_length, pooling='last_token_l2_v1',
            embedding_dim=self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, text):
        batch = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors='pt')
        if batch['input_ids'].shape[1] == 0:
            token = self.tokenizer.eos_token_id
            if token is None:
                raise ValueError('Encoder needs an EOS token for empty documents')
            batch = {'input_ids': torch.tensor([[token]]), 'attention_mask': torch.ones(1, 1, dtype=torch.long)}
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self.model(**batch, use_cache=False).last_hidden_state
        # One unpadded document at a time, including its final special token.
        return F.normalize(hidden[:, -1].float(), dim=-1)[0].cpu()
