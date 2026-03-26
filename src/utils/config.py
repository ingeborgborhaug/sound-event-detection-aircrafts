import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def _set_dotted(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _parse_cli_overrides(overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not overrides:
        return out

    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        _set_dotted(out, key.strip(), _parse_scalar(raw.strip()))
    return out


def load_config(
    default_path: str = "configs/default.yaml",
    experiment_path: Optional[str] = None,
    cli_overrides: Optional[Iterable[str]] = None,
) -> SimpleNamespace:
    default_file = Path(default_path)
    if not default_file.exists():
        raise FileNotFoundError(f"Default config not found: {default_file}")

    with default_file.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    if experiment_path:
        exp_file = Path(experiment_path)
        if not exp_file.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_file}")
        with exp_file.open("r", encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, exp_cfg)

    cli_cfg = _parse_cli_overrides(cli_overrides)
    cfg = _deep_merge(cfg, cli_cfg)

    return _to_ns(cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load merged project config")
    parser.add_argument("--config", type=str, default=None, help="Optional experiment config file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Dotted overrides, e.g. --set training.lr=0.0005",
    )
    return parser
