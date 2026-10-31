<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mark.svg">
    <img src="assets/mark.svg" alt="SynthStats" width="140">
  </picture>
</p>

# SynthStats

SynthStats trains language-model agents to write probabilistic programs.

## Install

```bash
git clone https://github.com/SynthStats/synthstats.git
cd synthstats
uv sync
```

## Example: show the dugongs training configuration

```bash
uv run synthstats-train env=dugongs --cfg job
```

## Contents

| Command or file | Purpose |
| --- | --- |
| `synthstats-train` | Train an agent |
| `src/synthstats/configs/train.yaml` | Default training configuration |
| `src/synthstats/train/gfn_core.py` | Shared trainer logic |

[Project](https://synthstats.org).

SynthStats is distributed under the [MIT licence](LICENSE).
