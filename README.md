<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mark.svg">
    <img src="assets/mark.svg" alt="SynthStats" width="140">
  </picture>
</p>

<p align="center">
  <a href="https://synthstats.org"><img src="https://img.shields.io/badge/website-synthstats.org-5C56A5?labelColor=5A6078" alt="Website: synthstats.org"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-MIT-5C56A5?labelColor=5A6078" alt="Licence: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%3E%3D3.12%2C%3C3.14-5C56A5?labelColor=5A6078" alt="Python: >=3.12,<3.14"></a>
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
