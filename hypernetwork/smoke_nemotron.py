"""Acceptance harness around the production recipe; no replacement training loop.

Train one stage with native -c/--dotted overrides, or compare a completed workflow:
python -m hypernetwork.smoke_nemotron --compare CHECKPOINT_ROOT
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from safetensors.torch import load_file
from nemo_automodel.components.checkpoint.utils import find_latest_checkpoint
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from .model import HyperNetwork
from .nemotron import HypernetworkDiffusionLMSFTRecipe


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_digest(tensor).encode())
    return digest.hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_digest(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def loader_state(loader):
    # TorchData restores lazily. state_dict() would instantiate the iterator,
    # whose sampler reports yielded=0 until its first next(), despite retaining
    # the correct resume position. Inspect the pending native state directly.
    pending = getattr(loader, 'next_iter_state', None)
    return pending if pending is not None else loader.state_dict()


def state_summary(recipe):
    steps = [int(v['step'].item()) for opt in recipe.optimizer for v in opt.state.values() if 'step' in v]
    return dict(step=recipe.step_scheduler.step, optimizer_steps=sorted(set(steps)),
                lr_scheduler=[plain(s.state_dict()) for s in recipe.lr_scheduler],
                dataloader=plain(loader_state(recipe.dataloader)))


def run_stage():
    cfg = parse_args_and_load_config()
    output = Path(cfg.checkpoint.checkpoint_dir)
    if output.exists():
        raise FileExistsError(f"Smoke stages require fresh output directories: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    recipe = HypernetworkDiffusionLMSFTRecipe(cfg)
    recipe.setup()
    model = recipe.model_parts[0]
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in model.backbone.parameters())
    assert all(p.dtype == torch.float32 and p.requires_grad for p in model.hypernetwork.parameters())
    before_base = model_digest(model.backbone)
    before_hyper = model_digest(model.hypernetwork)
    initial_state = state_summary(recipe)
    start = recipe.step_scheduler.step
    if cfg.get('checkpoint.restore_from', None):
        saved = Path(cfg.checkpoint.restore_from)
        scheduler = torch.load(saved / 'step_scheduler.pt', weights_only=True)
        assert start == scheduler['step'] == 5, scheduler
        assert initial_state['optimizer_steps'] == [start]
        loaders = list((saved / 'dataloader').glob('*.pt'))
        assert len(loaders) == 1, loaders
        restored_loader = torch.load(loaders[0], weights_only=False, map_location='cpu')
        assert initial_state['dataloader'] == plain(restored_loader), 'Dataloader state did not restore exactly'
        previous_report = json.loads((saved.parent / 'summary.json').read_text())
        assert initial_state['lr_scheduler'] == previous_report['final_state']['lr_scheduler'], 'LR state mismatch'
        weights = load_file(str(saved / 'model/hypernetwork.safetensors'))
        assert all(torch.equal(v.cpu(), weights[k]) for k, v in model.hypernetwork.state_dict().items())
    setup_seconds = time.perf_counter() - started
    train_started = time.perf_counter()
    recipe.run_train_validation_loop()
    training_seconds = time.perf_counter() - train_started
    final_state = state_summary(recipe)
    assert recipe.step_scheduler.step == cfg.step_scheduler.max_steps, final_state
    assert final_state['optimizer_steps'] == [cfg.step_scheduler.max_steps]
    assert all(value.dtype == torch.float32 for opt in recipe.optimizer for state in opt.state.values()
               for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq'))
    assert before_base == model_digest(model.backbone), 'Frozen backbone changed'
    assert before_hyper != model_digest(model.hypernetwork), 'Hypernetwork did not update'
    assert all(p.grad is None for p in model.backbone.parameters())
    latest = Path(find_latest_checkpoint(str(output)))
    clone = HyperNetwork(model.hypernetwork.config, model.hypernetwork.spec).to(recipe.dist_env.device)
    clone.load_state_dict(load_file(str(latest / 'model/hypernetwork.safetensors')))
    context = torch.ones(model.hypernetwork.config.embedding_dim, device=recipe.dist_env.device)
    with torch.no_grad():
        expected, actual = model.hypernetwork(context), clone(context)
        assert all(torch.equal(a, b) for key in expected for a, b in zip(expected[key], actual[key]))
    records = [json.loads(line) for line in (output / 'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(start, cfg.step_scheduler.max_steps))
    for row in records:
        assert all(math.isfinite(row[k]) for k in ('loss', 'dllm_loss', 'grad_norm', 'tps', 'mem'))
        assert row['loss'] > row['dllm_loss'] >= 0 and row['grad_norm'] > 0, row
        assert row['tokens_per_step'] == cfg.model.sequence_length * cfg.step_scheduler.global_batch_size
    assert any(r['dllm_loss'] > 0 for r in records)
    validation = [json.loads(line) for line in (output / 'validation.jsonl').read_text().splitlines()]
    assert validation and all(math.isfinite(r['val_loss']) for r in validation)
    assert validation[-1]['step'] == cfg.step_scheduler.max_steps - 1
    offline = list(output.glob('wandb/offline-run-*'))
    assert offline, 'Missing offline W&B run'
    report = dict(status='passed', checkpoint=str(latest.resolve()), start_step=start,
                  completed_steps=recipe.step_scheduler.step, accumulation=recipe.step_scheduler.grad_acc_steps,
                  backbone_unchanged=True, hypernetwork_updated=True, reload_verified=True,
                  backbone_sha256=before_base, initial_state=initial_state, final_state=final_state,
                  setup_seconds=setup_seconds, training_validation_checkpoint_seconds=training_seconds,
                  peak_gpu_gib=max(r['mem'] for r in records),
                  median_warm_step_seconds=statistics.median(r['tokens_per_step']/r['tps'] for r in records[2:])
                      if len(records) > 2 else None,
                  records=records, validation=validation, offline_wandb=[str(p) for p in offline])
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'completed_steps', 'checkpoint', 'peak_gpu_gib')}, indent=2))
    import wandb
    if wandb.run is not None:
        wandb.finish()


def compare(root):
    root = Path(root)
    reports = {stage: json.loads((root / stage / 'summary.json').read_text())
               for stage in ('smoke', 'resume', 'reference', 'accumulation')}
    resumed, reference = reports['resume'], reports['reference']
    assert resumed['start_step'] == 5 and resumed['completed_steps'] == reference['completed_steps'] == 6
    assert reports['accumulation']['accumulation'] == 4
    assert len({r['backbone_sha256'] for r in reports.values()}) == 1
    weights = [load_file(str(Path(r['checkpoint']) / 'model/hypernetwork.safetensors')) for r in (resumed, reference)]
    max_difference = 0.0
    for key in weights[0]:
        torch.testing.assert_close(weights[0][key], weights[1][key], rtol=1e-4, atol=1e-6)
        max_difference = max(max_difference, float((weights[0][key] - weights[1][key]).abs().max()))
    for key in ('loss', 'dllm_loss', 'grad_norm'):
        assert math.isclose(resumed['records'][-1][key], reference['records'][-1][key], rel_tol=1e-4, abs_tol=1e-6), key
    # Native LR state also stores the planned schedule length; constant LR makes
    # that 5-vs-6-step configuration difference irrelevant to this equivalence check.
    assert resumed['final_state']['optimizer_steps'] == reference['final_state']['optimizer_steps'] == [6]
    assert resumed['final_state']['dataloader'] == reference['final_state']['dataloader']
    report = dict(status='passed', resume_matches_reference=True, max_weight_difference=max_difference,
                  rtol=1e-4, atol=1e-6, stages={k: v['checkpoint'] for k, v in reports.items()})
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--compare':
        compare(sys.argv[2])
    else:
        run_stage()
