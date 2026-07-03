import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_simulation_summary(summary_path: Path) -> dict[str, Any]:
    """
    Load simulation_summary.json produced by the parallel simulation manager.
    """

    if not summary_path.exists():
        raise FileNotFoundError(f"Simulation summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_analysis_workflow(
    *,
    summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:

    logger.info("Starting analysis workflow")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_simulation_summary(summary_path)
    results = summary.get("results", [])

    if not isinstance(results, list):
        raise ValueError("Invalid simulation summary: 'results' must be a list.")

    total = len(results)
    success = sum(1 for r in results if r.get("success") is True)
    failed = total - success
    merger_count = sum(int(r.get("merger_count") or 0) for r in results)
    evolved_binary_count = sum(
        int(r.get("evolved_binary_count") or 0) for r in results
    )
    generation_count = sum(int(r.get("generation_count") or 0) for r in results)
    successful_outputs = [
        r.get("output_file")
        for r in results
        if r.get("success") is True and r.get("output_file")
    ]
    generation_files = [
        path
        for r in results
        if r.get("success") is True
        for path in (r.get("generation_files") or [r.get("output_file")])
        if path
    ]
    sample_manifests = [
        r.get("manifest_file")
        for r in results
        if r.get("success") is True and r.get("manifest_file")
    ]
    worker_metadata = [
        r.get("metadata_file")
        for r in results
        if r.get("metadata_file")
    ]
    failed_results = [
        {
            "model_index": r.get("model_index"),
            "sample_index": r.get("sample_index"),
            "smbh_mass": r.get("smbh_mass"),
            "error": r.get("error"),
        }
        for r in results
        if r.get("success") is not True
    ]

    analysis = {
        "total": total,
        "success": success,
        "failed": failed,
        "success_fraction": success / total if total > 0 else 0.0,
        "merger_count": merger_count,
        "evolved_binary_count": evolved_binary_count,
        "generation_count": generation_count,
        "successful_outputs": successful_outputs,
        "generation_files": generation_files,
        "sample_manifests": sample_manifests,
        "worker_metadata": worker_metadata,
        "combined_files": summary.get("combined_files", []),
        "failed_results": failed_results,
    }

    analysis_path = output_dir / "analysis_summary.json"

    with analysis_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    logger.info("Analysis workflow finished: %s", analysis_path)

    return analysis
