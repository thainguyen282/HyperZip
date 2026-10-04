"""Small-data regression checks for token-budgeted, disk-backed FineWeb preparation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

from .fineweb_data import manifest, load_arrow_documents

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'attempt'))
from prepare_fineweb_hypernetwork_900m import document_examples, prepare


class Tokenizer:
    eos_token_id = 11
    def encode(self, text, **kwargs):
        return [20 + (ord(c) % 30) for c in text][:kwargs['max_length']]


class FineWebDataTests(unittest.TestCase):
    def test_exact_budgets_masks_disjoint_splits_and_disk_loading(self):
        rows = [{'text': f'document {i} ' + 'abcdef ' * 5} for i in range(3000)]
        rows = [row for row in rows for _ in range(2)]  # Deliberate duplicates.
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            metadata = prepare(output, iter(rows), Tokenizer(), {'train': 3301, 'validation': 203})
            checked, digest = manifest(output, 1024, verify_files=True)
            self.assertEqual(metadata, checked)
            self.assertEqual(len(digest), 64)
            ids = {}
            from datasets import load_from_disk
            for split, budget in [('train', 3301), ('validation', 203)]:
                full = load_from_disk(str(output / split))
                dataset = load_arrow_documents(output, split)
                self.assertTrue(dataset.cache_files)
                self.assertEqual(sum(sum(r['attention_mask']) for r in dataset), budget)
                self.assertEqual(sum(sum(r['loss_mask']) for r in dataset), budget-len(dataset))
                ids[split] = set(full['document_id'])
                self.assertEqual(len(ids[split]),len(full))
                for row in dataset:
                    length = sum(row['attention_mask'])
                    self.assertGreaterEqual(length, 2)
                    self.assertEqual(len(row['input_ids']), 1024)
                    self.assertEqual(row['input_ids'][length-1], 11)
                    self.assertEqual(row['loss_mask'][0], 0)
                    self.assertFalse(any(row['loss_mask'][length:]))
            self.assertFalse(ids['train'] & ids['validation'])
            path = next((output / 'train').glob('*.arrow'))
            with path.open('ab') as handle:
                handle.write(b'bad')
            with self.assertRaisesRegex(ValueError, 'changed'):
                manifest(output,1024)

    def test_odd_budget_never_leaves_one_token_example(self):
        stats={k: {'tokens':0,'documents':0} for k in ('train','validation')}
        rows=[{'text':f'{i}x'} for i in range(2000)]
        with sqlite3.connect(':memory:') as connection:
            examples=list(document_examples(rows,Tokenizer(),{'train':7,'validation':3},1024,42,stats,connection))
        self.assertEqual(stats['train']['tokens'],7)
        self.assertEqual(stats['validation']['tokens'],3)
        self.assertTrue(all(sum(row['attention_mask']) >= 2 for _,row in examples))


if __name__ == '__main__':
    unittest.main()
