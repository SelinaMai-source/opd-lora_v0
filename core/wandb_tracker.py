from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_tracking_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
    tracking = output_cfg.get("tracking", {}) if isinstance(output_cfg.get("tracking", {}), dict) else {}
    env_mode = str(os.environ.get("WANDB_MODE", "")).strip()
    mode = str(tracking.get("wandb_mode", env_mode or "online")).strip() or "online"
    use_wandb = bool(tracking.get("use_wandb", False))
    if env_mode == "disabled":
        use_wandb = False
    return {
        "use_wandb": use_wandb,
        "project": str(tracking.get("wandb_project", "") or os.environ.get("WANDB_PROJECT", "lora-citb")).strip(),
        "entity": str(tracking.get("wandb_entity", "") or os.environ.get("WANDB_ENTITY", "")).strip(),
        "group": str(tracking.get("wandb_group", "") or cfg.get("experiment_name", "")).strip(),
        "name": str(tracking.get("wandb_run_name", "") or output_cfg.get("run_name", "")).strip(),
        "tags": _normalize_tags(tracking.get("wandb_tags", [])),
        "mode": mode,
        "log_artifacts": bool(tracking.get("log_artifacts", True)),
        "notes": str(tracking.get("wandb_notes", "")).strip(),
    }


def _normalize_tags(tags: Any) -> List[str]:
    if isinstance(tags, list):
        return [str(x).strip() for x in tags if str(x).strip()]
    if isinstance(tags, str) and tags.strip():
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return ""


def _flatten_numeric(prefix: str, obj: Dict[str, Any], out: Dict[str, float]) -> None:
    for key, value in obj.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            out[full_key] = float(value)
        elif isinstance(value, (int, float)):
            out[full_key] = float(value)
        elif isinstance(value, dict):
            _flatten_numeric(full_key, value, out)


class WandbTracker:
    """Optional Weights & Biases logger; no-op when disabled or wandb is missing."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._run = None

    @classmethod
    def from_config(
        cls,
        *,
        cfg: Dict[str, Any],
        run_id: str,
        config_path: str,
        run_dir: str,
    ) -> "WandbTracker":
        tracking = parse_tracking_cfg(cfg)
        if not tracking["use_wandb"]:
            return cls(enabled=False)

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B tracking is enabled but `wandb` is not installed. "
                "Run: pip install wandb"
            ) from exc

        repo_root = Path(__file__).resolve().parents[1]
        tags = list(tracking["tags"])
        tags.extend(
            [
                str(cfg.get("mode", "")),
                f"seed:{cfg.get('seed', '')}",
            ]
        )
        paper_cfg = cfg.get("paper", {}) if isinstance(cfg.get("paper", {}), dict) else {}
        if paper_cfg.get("method_variant"):
            tags.append(str(paper_cfg["method_variant"]))
        tags = [tag for tag in tags if str(tag).strip()]

        init_kwargs: Dict[str, Any] = {
            "project": tracking["project"],
            "name": tracking["name"] or run_id,
            "group": tracking["group"] or None,
            "tags": tags or None,
            "config": cfg,
            "dir": str(Path(run_dir)),
            "mode": tracking["mode"],
            "notes": tracking["notes"] or None,
        }
        settings_cls = getattr(wandb, "Settings", None)
        if settings_cls is not None:
            init_kwargs["settings"] = settings_cls(start_method="thread")
        if tracking["entity"]:
            init_kwargs["entity"] = tracking["entity"]

        run = wandb.init(**init_kwargs)
        run.summary["run_id"] = run_id
        run.summary["config_path"] = config_path
        run.summary["git_commit"] = _git_head(repo_root)
        tracker = cls(enabled=True)
        tracker._run = run
        return tracker

    def log_segment_row(self, row: Dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb

        step = int(row.get("segment_id", 0))
        metrics: Dict[str, float] = {}
        for key, value in row.items():
            if key in {"segment_name", "run_id", "mode", "baseline_name", "active_adapter"}:
                continue
            if isinstance(value, bool):
                metrics[str(key)] = float(value)
            elif isinstance(value, (int, float)):
                metrics[str(key)] = float(value)
            elif isinstance(value, str) and key.endswith("_json"):
                continue
        wandb.log(metrics, step=step)

    def log_final(self, final_doc: Dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb

        summary: Dict[str, float] = {}
        final = final_doc.get("final", {}) if isinstance(final_doc.get("final", {}), dict) else {}
        drift_quality = final_doc.get("drift_quality", {}) if isinstance(final_doc.get("drift_quality", {}), dict) else {}
        routing_quality = (
            final_doc.get("routing_quality", {}) if isinstance(final_doc.get("routing_quality", {}), dict) else {}
        )
        _flatten_numeric("final", final, summary)
        _flatten_numeric("drift_quality", drift_quality, summary)
        _flatten_numeric("routing_quality", routing_quality, summary)
        for key, value in summary.items():
            wandb.run.summary[key] = value

    def log_artifacts(self, paths: List[Path]) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb

        for path in paths:
            if path.is_file():
                wandb.save(str(path), base_path=str(path.parent))

    def finish(self, *, success: bool, error: str = "") -> None:
        if not self.enabled or self._run is None:
            return
        import wandb

        wandb.run.summary["success"] = bool(success)
        if error:
            wandb.run.summary["error"] = error
        wandb.finish(exit_code=0 if success else 1)
