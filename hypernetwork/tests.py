"""Offline integration tests: gradients, PEFT equivalence, and real archives."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from config import parse_args
from compression.lora import inspect_lora, load_lora
from compression.pipeline import HYPERNETWORK_FORMAT, read_archive
from .checkpoint import (save_checkpoint, load_checkpoint, generate, export_adapter,
                         embedding_bytes, read_embedding)
from .model import HyperConfig, HyperNetwork, generated_lora, target_spec
from .train import masked_diffusion_loss
import main


class ByteTokenizer:
    bos_token_id = 0
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))

    def decode(self, symbols, **kwargs):
        return bytes(symbols).decode('utf-8')


class HypernetworkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.manual_seed(42)
        self.base = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=64, bos_token_id=0, eos_token_id=1,
            attention_dropout=0.0)).eval().requires_grad_(False)
        self.base_path = self.root / 'base'
        self.base.save_pretrained(self.base_path)
        self.network = HyperNetwork(HyperConfig(embedding_dim=6, identity_dim=3,
            hidden_dim=12, rank=2, alpha=4), target_spec(self.base))
        self.context = torch.randn(6)

    def learn(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.02)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        for _ in range(3):
            optimizer.zero_grad()
            with generated_lora(self.base, self.network(self.context), 4, 2):
                loss = self.base(ids, labels=ids).loss
                loss.backward()
            optimizer.step()
        return ids

    def checkpoint(self):
        path = self.root / 'hyper'
        save_checkpoint(path, self.network,
            backbone=dict(model='autoregressive', model_path=str(self.base_path), revision=None),
            encoder=dict(model_path='not-needed-for-decoding', max_length=32,
                         embedding_dim=6, pooling='last_token_l2_v1'))
        return path

    def encode(self, raw=b'hello\r\n', computed_context=False):
        self.learn()
        checkpoint = self.checkpoint()
        source = self.root / 'source.txt'
        source.write_bytes(raw)
        context = self.root / 'context.npy'
        np.save(context, self.context.numpy())
        argv = ['encode', '--input', str(source), '--output', str(self.root / 'archive'),
            '--model_path', str(self.base_path), '--hypernetwork_path', str(checkpoint),
            '--device', 'cpu', '--dtype', 'float32', '--local_files_only', '--no_progress',
            '--context_tokens', '8']
        if not computed_context:
            argv.extend(['--context_embedding', str(context)])
        args = parse_args(argv)
        with patch('compression.model_loading._tokenizer', return_value=ByteTokenizer()):
            metrics = main.run(args)
        return metrics

    def decode(self, *extra):
        args = parse_args(['decode', '--input', str(self.root / 'archive'),
            '--output', str(self.root / 'restored'), '--device', 'cpu', '--local_files_only',
            '--no_progress', *extra])
        with patch('compression.model_loading._tokenizer', return_value=ByteTokenizer()):
            return main.run(args)

    def test_training_changes_only_hypernetwork_and_reaches_conditioning(self):
        before = {k: v.clone() for k, v in self.base.state_dict().items()}
        ids = self.learn()
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.base.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        self.assertGreater(self.network.layers.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.network.modules_embedding.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            plain = self.base(ids).logits
            with generated_lora(self.base, self.network(self.context), 4, 2):
                adapted = self.base(ids).logits
            with generated_lora(self.base, self.network(-self.context), 4, 2):
                other = self.base(ids).logits
        self.assertFalse(torch.equal(plain, adapted))
        self.assertFalse(torch.equal(adapted, other))
        self.assertTrue(torch.equal(plain, self.base(ids).logits))

    def test_export_matches_peft_and_checkpoint_regeneration(self):
        ids = self.learn()
        checkpoint = self.checkpoint()
        loaded, _, _ = load_checkpoint(checkpoint)
        context = read_embedding(embedding_bytes(self.context.numpy()), 6)
        a, digest = generate(self.network, context)
        b, other_digest = generate(loaded, context)
        self.assertEqual(digest, other_digest)
        for name in a:
            self.assertTrue(torch.equal(a[name][0], b[name][0]))
            self.assertTrue(torch.equal(a[name][1], b[name][1]))
        with generated_lora(self.base, a, 4, 2), torch.no_grad():
            expected = self.base(ids).logits
        export_adapter(self.root / 'adapter', loaded, context, str(self.base_path))
        peft = load_lora(self.base, inspect_lora(self.root / 'adapter'))
        with torch.no_grad():
            actual = peft(ids).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_archive_roundtrip_without_source_encoder_or_adapter(self):
        original = 'a🙂é\r\n漢字'.encode()
        metrics = self.encode(original)
        archive = read_archive(self.root / 'archive')
        self.assertEqual(archive.header['format'], HYPERNETWORK_FORMAT)
        self.assertEqual(len(archive.context), 6 * 4)
        self.assertEqual(metrics['compressed_bytes'], (self.root / 'archive').stat().st_size)
        (self.root / 'source.txt').unlink()
        (self.root / 'context.npy').unlink()
        relocated = self.root / 'relocated'
        (self.root / 'hyper').rename(relocated)
        with patch('hypernetwork.runtime.DocumentEncoder', side_effect=AssertionError('Decoder embedded source')):
            result = self.decode('--hypernetwork_path', str(relocated))
        self.assertTrue(result['verified_decode'])
        self.assertEqual((self.root / 'restored').read_bytes(), original)

    def test_empty_document_roundtrip(self):
        self.encode(b'')
        self.assertTrue(self.decode()['verified_decode'])
        self.assertEqual((self.root / 'restored').read_bytes(), b'')

    def test_encode_embeds_once_and_decode_never_loads_encoder(self):
        with patch('hypernetwork.runtime.DocumentEncoder') as encoder:
            encoder.return_value.encode.return_value = self.context
            metrics = self.encode(computed_context=True)
            encoder.assert_called_once()
            encoder.return_value.encode.assert_called_once_with('hello\r\n')
        self.assertEqual(metrics['context_bytes'], 24)
        self.assertGreater(metrics['personalization_seconds'], 0)
        with patch('hypernetwork.runtime.DocumentEncoder', side_effect=AssertionError('Unexpected encoder')):
            self.assertTrue(self.decode()['verified_decode'])

    def test_runtime_is_configured_before_document_encoder_initializes_cuda(self):
        from compression.model_loading import configure_runtime
        encoder = Mock()
        encoder.encode.return_value = self.context
        with patch('hypernetwork.runtime.configure_runtime', wraps=configure_runtime) as configure:
            def create_encoder(*args, **kwargs):
                configure.assert_called_once()
                return encoder
            with patch('hypernetwork.runtime.DocumentEncoder', side_effect=create_encoder):
                self.encode(computed_context=True)

    def test_corrupt_context_rejected_before_model_loading(self):
        self.encode()
        archive = self.root / 'archive'
        with ZipFile(archive) as stream:
            members = {name: stream.read(name) for name in stream.namelist()}
        members['context.f32'] = bytes(24)
        with ZipFile(archive, 'w', compression=ZIP_STORED) as stream:
            for name, value in members.items():
                stream.writestr(name, value)
        with patch('main.load_model') as loader, self.assertRaisesRegex(ValueError, 'Context vector checksum'):
            self.decode()
        loader.assert_not_called()

    def test_changed_checkpoint_rejected_before_model_loading(self):
        self.encode()
        file = self.root / 'hyper/config.json'
        file.write_text(file.read_text() + ' ')
        with patch('main.load_model') as loader, self.assertRaisesRegex(ValueError, 'checkpoint checksum'):
            self.decode()
        loader.assert_not_called()

    def test_bad_embedding_dimensions_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            self.network(torch.ones(7))
        with self.assertRaises(ValueError):
            embedding_bytes(np.array([np.nan]))
        with self.assertRaises(ValueError):
            read_embedding(bytes(20), 6)

    def test_generated_hooks_are_removed_on_failure(self):
        with self.assertRaises(RuntimeError):
            with generated_lora(self.base, self.network(self.context), 4, 2):
                raise RuntimeError('interrupted')
        self.assertTrue(all(not m._forward_hooks for m in self.base.modules()))

    def test_prepare_cli_uses_frozen_encoder_and_persists_full_token_sequence(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from .__main__ import main as hyper_main
        tokenizer = Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, 'hello': 2, 'world': 3}, unk_token='<unk>'))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                            unk_token='<unk>', eos_token='<eos>')
        tokenizer.save_pretrained(self.base_path)
        corpus = self.root / 'corpus.jsonl'
        corpus.write_text(json.dumps({'text': 'hello world hello world'}) + '\n')
        prepared = self.root / 'prepared'
        hyper_main(['prepare', '--input', str(corpus), '--output', str(prepared),
            '--tokenizer_path', str(self.base_path), '--embedding_model', str(self.base_path),
            '--encoder_max_length', '2', '--device', 'cpu', '--local_files_only'])
        record = json.loads((prepared / 'documents.jsonl').read_text())
        metadata = json.loads((prepared / 'metadata.json').read_text())
        self.assertEqual(record['input_ids'], [2, 3, 2, 3])
        self.assertEqual(len(record['embedding']), 16)
        self.assertAlmostEqual(float(np.linalg.norm(record['embedding'])), 1.0, places=6)
        self.assertEqual(metadata['encoder']['max_length'], 2)

    def test_adapter_fingerprint_mismatch_rejected_before_loading_backbone(self):
        self.encode()
        path = self.root / 'archive'
        with ZipFile(path) as stream:
            members = {name: stream.read(name) for name in stream.namelist()}
        header = json.loads(members['manifest.json'])
        header['hypernetwork']['adapter_sha256'] = '0' * 64
        members['manifest.json'] = json.dumps(header).encode()
        with ZipFile(path, 'w', compression=ZIP_STORED) as stream:
            for name, value in members.items():
                stream.writestr(name, value)
        with patch('main.load_model') as loader, self.assertRaisesRegex(ValueError, 'adapter checksum'):
            self.decode()
        loader.assert_not_called()

    def test_checkpointed_gradients_match_regular_gradients(self):
        from .train import checkpoint_layers
        ids = torch.tensor([[1, 2, 3, 4]])
        with generated_lora(self.base, self.network(self.context), 4, 2):
            self.base(ids, labels=ids, use_cache=False).loss.backward()
        expected = {name: p.grad.clone() for name, p in self.network.named_parameters()}
        self.network.zero_grad(set_to_none=True)
        with generated_lora(self.base, self.network(self.context), 4, 2), checkpoint_layers(self.base):
            self.base(ids, labels=ids, use_cache=False).loss.backward()
        for name, parameter in self.network.named_parameters():
            torch.testing.assert_close(parameter.grad, expected[name], rtol=0, atol=0)

    def test_masked_loss_aligns_next_position_and_hides_targets(self):
        from types import SimpleNamespace
        class Capture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sentinel = torch.nn.Parameter(torch.zeros(()))
            def forward(self, input_ids, logits_to_keep, **kwargs):
                self.ids = input_ids
                self.logits = torch.randn(1, logits_to_keep, 256, requires_grad=True)
                return SimpleNamespace(logits=self.logits)
        model = Capture()
        tokens = [10, 11, 12, 13, 14, 15, 16, 17]
        loss = masked_diffusion_loss(model, tokens, eos=1, mask=255, block_size=4,
                                     sequence_length=12, rng=random.Random(3))
        positions = (model.ids[0] == 255).nonzero().flatten()
        start = (model.ids.shape[1] - 1) // 4 * 4
        self.assertTrue((positions > start).all())
        self.assertTrue(torch.equal(model.ids[0, :start], torch.tensor([1] + tokens)[:start]))
        expected = torch.nn.functional.cross_entropy(model.logits[0, positions - start - 1],
                                                     torch.tensor([1] + tokens)[positions])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(model.logits.grad)


if __name__ == '__main__':
    unittest.main()
