# Training

Training is isolated under `action/training`; it trains parameter policies and
does not define Skills, terminal criteria, robot models, or environments.

## Independent BC

`TrainingSample` contains one Skill, one finite observation tuple, and one
five-dimensional `RobotAction`. `ActionDataset` rejects empty data, mixed
Skills, or mixed observation dimensions. `BehaviorCloningTrainer` builds a
small injected MLP, trains MSE with a `tanh` output, and exports TorchScript.

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python scripts\train\train_skill_bc.py `
  --skill ALIGN `
  --dataset path\to\align_samples.json `
  --output outputs\align\policy.ts
```

The JSON input is either a list or `{"samples": [...]}`. Every row contains
`observation` and `action` arrays. This entry trains one Skill only.

## Registry and checkpoints

`TrainerRegistry` keys trainers by `(Skill, TrainingAlgorithm)` and fails on an
unregistered pair. Algorithms are `BC`, `DAGGER`, `PPO`, and `STAGE_PPO`.
`CheckpointRecord` stores the exported artifact metadata; runtime hashes belong
in `config/artifacts.lock.json`.

## Retained V5 workflows

The validated StagePPO and DAgger implementations still depend on the V5 Isaac
runtime. `legacy_stageppo_command` creates an explicit command targeting the
vendored training script and validates both the Isaac Python executable and
script path. It never silently invokes legacy training.

No training was run during this structural refactor. Existing V5-trained
TorchScript policies remain unchanged. The compatibility verifier proves that
their routed V7 outputs match direct legacy inference after the same frozen
safety projection.

## Ownership constraints

- Skill meaning and transitions: `action/actions` and `action/action_list.py`.
- Success and failure: `action/evaluation`.
- Policy network and artifact loading: `action/network`.
- Robot, scene, and physics: `simulation`.
- Training loop and dataset: `action/training` only.
