from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


def build_sft_record(code: str):
    import sft_to_atif

    return sft_to_atif.build_atif_record(
        0,
        "Write a model.",
        f'<submit_program language="haskell">{code}</submit_program>',
        ByteTokenizer(),
    )


def load_script_module(module_name: str):
    script_path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"testscripts_{module_name}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module
