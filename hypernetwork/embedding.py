"""Frozen document encoder; last-token pooling and L2 normalization (GTE)."""
import torch
from torch.nn import functional as F

DEFAULT_ENCODER = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'


def project_context(vector, dimension):
    """Legacy fixed projection for checkpoints prepared before linear-only training."""
    if dimension < 1 or dimension > vector.numel():
        raise ValueError('Context dimension must be between 1 and encoder width')
    if dimension == vector.numel():
        return vector
    pooled = F.adaptive_avg_pool1d(vector.float().reshape(1, 1, -1), dimension)[0, 0]
    return F.normalize(pooled, dim=0)


class DocumentEncoder:
    def __init__(self, model_path=DEFAULT_ENCODER, *, revision=None, max_length=32000,
                 device='cpu', dtype='float32', trust_remote_code=False, local_files_only=False,
                 pooling='last_token'):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        if max_length < 1:
            raise ValueError('Encoder max_length must be positive')
        if pooling not in ('last_token', 'mean'):
            raise ValueError('pooling must be last_token or mean')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision,
            trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        config = AutoConfig.from_pretrained(model_path, revision=revision,
            trust_remote_code=trust_remote_code, local_files_only=local_files_only)
        # The GTE-Qwen2 remote model code still reads the pre-Transformers-5 attribute.
        if model_path == DEFAULT_ENCODER and not hasattr(config, 'rope_theta'):
            config.rope_theta = config.rope_parameters['rope_theta']
        self.model = AutoModel.from_pretrained(model_path, revision=revision,
            trust_remote_code=trust_remote_code, local_files_only=local_files_only,
            config=config, dtype=getattr(torch, dtype)).to(device).eval().requires_grad_(False)
        self.device, self.max_length, self.pooling = device, max_length, pooling
        self.metadata = dict(model_path=model_path,
            revision=getattr(self.model.config, '_commit_hash', None) or revision,
            max_length=max_length, pooling=f'{pooling}_l2_v1',
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
        hidden = self.model(**batch, use_cache=False).last_hidden_state.float()
        if self.pooling == 'mean':
            mask = batch['attention_mask'].unsqueeze(-1)
            vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            # One unpadded document at a time, including its final special token.
            vector = hidden[:, -1]
        return F.normalize(vector, dim=-1)[0].cpu()
