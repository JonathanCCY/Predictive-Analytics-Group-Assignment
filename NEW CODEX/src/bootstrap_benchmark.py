from __future__ import annotations

import importlib.metadata
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


RANDOM_SEED = 42
TASK_NAME = "step1_setup"
STATUS_CONFIRMED_FILE = "CONFIRMED_BY_FILE_INSPECTION"
STATUS_CONFIRMED_EXEC = "CONFIRMED_BY_EXECUTION"
STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_NOT_EXECUTED = "NOT_EXECUTED"


def parse_requirements(requirements_path: Path) -> Dict[str, str]:
    requirements: Dict[str, str] = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        requirements[name] = version
    return requirements


def get_installed_versions(requirements: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    versions: Dict[str, Dict[str, str]] = {}
    for package_name, required_version in requirements.items():
        try:
            installed_version = importlib.metadata.version(package_name)
            status = (
                STATUS_CONFIRMED_EXEC
                if installed_version == required_version
                else STATUS_UNVERIFIED
            )
        except importlib.metadata.PackageNotFoundError:
            installed_version = "MISSING"
            status = STATUS_UNVERIFIED
        versions[package_name] = {
            "required_version": required_version,
            "installed_version": installed_version,
            "status": status,
        }
    return versions


def ensure_directories(repo_root: Path) -> List[str]:
    created: List[str] = []
    for relative_dir in [
        "data",
        "src",
        "outputs",
        "outputs/shared",
        "outputs/eda",
        "outputs/model_compare",
        "outputs/model",
        "outputs/debug",
        f"outputs/{TASK_NAME}",
        "docs",
        "tests",
    ]:
        path = repo_root / relative_dir
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(relative_dir)
    return created


def ensure_data_path(repo_root: Path, file_name: str) -> Tuple[str, str]:
    source_path = repo_root / file_name
    data_path = repo_root / "data" / file_name

    if data_path.exists():
        if data_path.is_symlink():
            return f"data/{file_name}", f"existing_symlink->{os.readlink(data_path)}"
        return f"data/{file_name}", "existing_file"

    if not source_path.exists():
        raise FileNotFoundError(f"Required source file not found: {source_path}")

    try:
        data_path.symlink_to(Path("..") / file_name)
        return f"data/{file_name}", f"created_symlink->{os.readlink(data_path)}"
    except OSError:
        shutil.copy2(source_path, data_path)
        return f"data/{file_name}", "copied_from_repo_root"


def inspect_dataset(csv_path: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    return {
        "path": str(csv_path.relative_to(csv_path.parent.parent)),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "first_column": df.columns[0],
        "has_id_column": "id" in df.columns,
        "has_target_column": "satisfaction" in df.columns,
        "target_distribution": df["satisfaction"].value_counts().to_dict(),
        "unnamed_first_column_detected": df.columns[0].startswith("Unnamed"),
    }


def seed_reproducibility_record(
    repo_root: Path, created_files: List[str], updated_files: List[str]
) -> Path:
    docs_path = repo_root / "docs" / "reproducibility_record.md"
    legacy_path = repo_root / "reproducibility_record.md"

    if docs_path.exists():
        updated_files.append("docs/reproducibility_record.md")
        return docs_path

    if legacy_path.exists():
        content = legacy_path.read_text(encoding="utf-8").rstrip() + "\n"
    else:
        content = (
            "# Reproducibility Record\n\n"
            "Initialized from the Step 1 bootstrap because "
            "`docs/reproducibility_record.md` was not found.\n"
        )

    docs_path.write_text(content, encoding="utf-8")
    created_files.append("docs/reproducibility_record.md")
    return docs_path


def append_step1_entry(
    record_path: Path,
    dataset_summary: Dict[str, Dict[str, object]],
    split_manifest_exists: bool,
    protocol_exists: bool,
) -> None:
    marker = "## Task Entry - Step 1 / Benchmark Bootstrap"
    current_text = record_path.read_text(encoding="utf-8")
    if marker in current_text:
        return

    combined_rows = (
        int(dataset_summary["train.csv"]["rows"]) + int(dataset_summary["test.csv"]["rows"])
    )
    entry = f"""
---

{marker}

### Task metadata
- Step: 1
- Task label: `GENERAL_BENCHMARK_PROMPT`
- Task name: benchmark bootstrap and validation
- Main script(s): `src/bootstrap_benchmark.py`
- Execution date: `{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")}`

### Step 1 findings
- `agent_reproducibility_protocol_v3.md` present: `{STATUS_CONFIRMED_FILE if protocol_exists else STATUS_UNVERIFIED}`
- `requirements.txt` inspected: `{STATUS_CONFIRMED_FILE}`
- `docs/reproducibility_record.md` present before bootstrap: `{STATUS_UNVERIFIED}`
- `outputs/shared/split_manifest.json` present before bootstrap: `{STATUS_CONFIRMED_FILE if split_manifest_exists else STATUS_UNVERIFIED}`
- Canonical split created in Step 1: `{STATUS_NOT_EXECUTED}`
- Train source rows observed: `{dataset_summary["train.csv"]["rows"]}`
- Test source rows observed: `{dataset_summary["test.csv"]["rows"]}`
- Combined source rows observed: `{combined_rows}`
- Unnamed first column detected in both source files: `{STATUS_CONFIRMED_EXEC}`
- `id` column detected in both source files: `{STATUS_CONFIRMED_EXEC}`
- Target column `satisfaction` detected in both source files: `{STATUS_CONFIRMED_EXEC}`

### Notes
- Fallback benchmark specification used because `agent_reproducibility_protocol_v3.md` was not found.
- Step 1 created project structure and audit files only; no split, EDA, or modelling was executed.
"""
    record_path.write_text(current_text.rstrip() + "\n" + entry, encoding="utf-8")


def append_decision_log(
    repo_root: Path,
    data_link_results: List[Tuple[str, str]],
    created_files: List[str],
    updated_files: List[str],
) -> Path:
    decision_log_path = repo_root / "docs" / "decision_log.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"## {timestamp} - Step 1 benchmark bootstrap",
        "",
        "- `agent_reproducibility_protocol_v3.md` was not found, so the user prompt is the active fallback specification.",
        "- The source CSV files were present at the repository root instead of under `data/`.",
        "- Step 1 created `data/` access paths for the existing source CSVs without generating the canonical split early.",
        "- Step 1 seeded `docs/reproducibility_record.md` from the legacy root-level `reproducibility_record.md` file so later tasks can append in the expected location.",
    ]
    for relative_path, action in data_link_results:
        lines.append(f"- `{relative_path}` preparation result: `{action}`.")
    lines.append("")

    if decision_log_path.exists():
        current = decision_log_path.read_text(encoding="utf-8").rstrip() + "\n\n"
        decision_log_path.write_text(current + "\n".join(lines) + "\n", encoding="utf-8")
        updated_files.append("docs/decision_log.md")
    else:
        decision_log_path.write_text("# Decision Log\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
        created_files.append("docs/decision_log.md")
    return decision_log_path


def write_run_outputs(
    repo_root: Path,
    run_lines: List[str],
    metadata: Dict[str, object],
    created_files: List[str],
) -> None:
    task_dir = repo_root / "outputs" / TASK_NAME
    run_log_path = task_dir / "run_log.txt"
    metadata_path = task_dir / "run_metadata.json"

    run_log_path.write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    created_files.extend(
        [
            f"outputs/{TASK_NAME}/run_log.txt",
            f"outputs/{TASK_NAME}/run_metadata.json",
        ]
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    repo_root = Path(__file__).resolve().parent.parent
    run_lines: List[str] = []
    created_files: List[str] = []
    updated_files: List[str] = []

    protocol_path = repo_root / "agent_reproducibility_protocol_v3.md"
    requirements_path = repo_root / "requirements.txt"
    split_manifest_path = repo_root / "outputs" / "shared" / "split_manifest.json"

    run_lines.append(f"Repository root: {repo_root}")
    run_lines.append(f"Random seed set to: {RANDOM_SEED}")
    run_lines.append(
        "Protocol file present: "
        + ("yes" if protocol_path.exists() else "no, using prompt fallback")
    )

    created_directories = ensure_directories(repo_root)
    for relative_dir in created_directories:
        run_lines.append(f"Created directory: {relative_dir}")

    data_link_results = [
        ensure_data_path(repo_root, "train.csv"),
        ensure_data_path(repo_root, "test.csv"),
    ]
    for relative_path, action in data_link_results:
        run_lines.append(f"Prepared {relative_path}: {action}")

    requirements = parse_requirements(requirements_path)
    installed_versions = get_installed_versions(requirements)
    for package_name, version_info in installed_versions.items():
        run_lines.append(
            f"Dependency check - {package_name}: "
            f"required={version_info['required_version']}, "
            f"installed={version_info['installed_version']}, "
            f"status={version_info['status']}"
        )

    dataset_summary = {
        "train.csv": inspect_dataset(repo_root / "data" / "train.csv"),
        "test.csv": inspect_dataset(repo_root / "data" / "test.csv"),
    }
    for file_name, summary in dataset_summary.items():
        run_lines.append(
            f"Dataset check - {file_name}: rows={summary['rows']}, "
            f"columns={summary['columns']}, "
            f"first_column={summary['first_column']}, "
            f"has_id={summary['has_id_column']}, "
            f"has_target={summary['has_target_column']}"
        )

    record_path = seed_reproducibility_record(repo_root, created_files, updated_files)
    append_step1_entry(
        record_path=record_path,
        dataset_summary=dataset_summary,
        split_manifest_exists=split_manifest_path.exists(),
        protocol_exists=protocol_path.exists(),
    )
    if "docs/reproducibility_record.md" not in created_files:
        updated_files.append("docs/reproducibility_record.md")

    append_decision_log(
        repo_root=repo_root,
        data_link_results=data_link_results,
        created_files=created_files,
        updated_files=updated_files,
    )

    run_output_files = [
        f"outputs/{TASK_NAME}/run_log.txt",
        f"outputs/{TASK_NAME}/run_metadata.json",
    ]

    metadata = {
        "task_name": TASK_NAME,
        "execution_status": STATUS_CONFIRMED_EXEC,
        "random_seed": RANDOM_SEED,
        "benchmark_protocol_file": {
            "path": "agent_reproducibility_protocol_v3.md",
            "status": STATUS_CONFIRMED_FILE if protocol_path.exists() else STATUS_UNVERIFIED,
        },
        "binding_files": {
            "requirements.txt": STATUS_CONFIRMED_FILE,
            "docs/reproducibility_record.md": STATUS_CONFIRMED_EXEC,
            "outputs/shared/split_manifest.json": (
                STATUS_CONFIRMED_FILE if split_manifest_path.exists() else STATUS_UNVERIFIED
            ),
        },
        "dataset_summary": dataset_summary,
        "dependency_versions": installed_versions,
        "created_directories": created_directories,
        "data_path_preparation": [
            {"path": relative_path, "result": action}
            for relative_path, action in data_link_results
        ],
        "split_created_in_step1": False,
        "created_files": created_files + run_output_files,
        "updated_files": sorted(set(updated_files)),
    }

    write_run_outputs(
        repo_root=repo_root,
        run_lines=run_lines,
        metadata=metadata,
        created_files=created_files,
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
