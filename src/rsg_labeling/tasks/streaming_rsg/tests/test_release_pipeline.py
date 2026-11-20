"""Generation release smoke checks isolated from other projects' Python state."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "command",
    [[], ["run"], ["prepare-video"], ["export-public"], ["export-annotation"], ["parallel"]],
)
def test_cli_help_without_runtime_dependencies(command, tmp_path):
    """Help remains usable before installing optional dataset and inference dependencies."""
    package = Path(__file__).absolute().parents[3]
    module = f"{__package__.split('.')[0]}.tasks.streaming_rsg.generation"
    args = command
    if command == ["parallel"]:
        module += ".parallel"
        args = []
    result = subprocess.run(
        [sys.executable, "-S", "-m", module, *args, "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(package.parent)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("command", ["run", "prepare-video", "parallel"])
def test_missing_config_is_a_cli_usage_error(command, tmp_path):
    """A missing local config fails before runtime imports, output creation or HTTP."""
    package = Path(__file__).absolute().parents[3]
    module = f"{__package__.split('.')[0]}.tasks.streaming_rsg.generation"
    args = [command]
    if command == "parallel":
        module += ".parallel"
        args = ["--workers", "2"]
    elif command == "prepare-video":
        args += ["--videos", str(tmp_path)]
    missing = tmp_path / "missing.yaml"
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-S", "-m", module, *args, "--config", str(missing), "--out", str(output)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(package.parent)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert f"generation config file does not exist: {missing}" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_unfilled_setup_names_fields_and_preserves_preparation_and_replay(tmp_path):
    """Unset template fields fail together without requiring generation-only preparation inputs."""
    configuration = importlib.import_module(
        f"{__package__.split('.')[0]}.tasks.streaming_rsg.generation.configuration"
    )
    config = {
        "mode": "native",
        "compiler_config": None,
        "endpoint_registry": None,
        "input_manifest": None,
        "allow_paid_dispatch": False,
        "budget": {},
    }
    with pytest.raises(ValueError) as error:
        configuration.validate_setup(config)
    for field in (
        "compiler_config",
        "endpoint_registry",
        "input_manifest",
        "allow_paid_dispatch",
        "budget.ledger",
        "budget.owner",
        "budget.policy",
    ):
        assert field in str(error.value)
    config.update(compiler_config="compiler.yaml", endpoint_registry="registry")
    configuration.validate_setup(config, preparation=True)
    config.update(
        mode="frozen_graph",
        compiler_config=None,
        input_manifest="inputs.json",
        source_run="source",
        allow_paid_dispatch=True,
        budget={
            "owner": "owner",
            "campaign": "campaign",
            "ledger": "ledger.jsonl",
            "accounting_verified_at": "2026-11-12T00:00:00Z",
            "dispatch_envelope_usd": 1,
            "policy": {},
        },
    )
    configuration.validate_setup(config)
    config["input_manifest"] = "/absolute/path/to/inputs.json"
    with pytest.raises(ValueError, match="input_manifest"):
        configuration.validate_setup(config)


@pytest.mark.parametrize(
    "mode",
    [
        "native",
        "parallel",
        "video",
        "legacy",
        "transient",
        "malformed",
        "provider",
        "missing-helper",
        "archive-rejected",
    ],
)
def test_generation_resume_and_public_export(mode, tmp_path):
    """Run from only selected sources; an omitted helper cannot fall back to this checkout."""
    package = Path(__file__).resolve().parents[3]
    if not (package / "infra/budget.py").is_file():
        pytest.skip(
            "Folder-only smoke requires the materialized release; use the maintainer checker"
        )
    manifest = json.loads((package / "releases/files.json").read_text())
    root = tmp_path / "selected"
    for relative in manifest["files"]:
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        target = root / "rsg_labeling" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, target)
    if mode == "missing-helper":
        (root / "rsg_labeling/common/coc.py").unlink()
    command = [
        sys.executable,
        "-m",
        "rsg_labeling.tasks.streaming_rsg.tests.release_smoke",
        "native" if mode == "missing-helper" else mode,
    ]
    if mode == "provider":
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--pyargs",
            "rsg_labeling.tasks.streaming_rsg.tests.test_openai_profile",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
    result = subprocess.run(
        command,
        check=False,
        cwd=root,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    if mode == "missing-helper":
        assert result.returncode != 0, "Missing selected helper was borrowed from dev"
        assert "rsg_labeling.common.coc" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert ("passed" if mode == "provider" else "dependency isolation passed") in result.stdout
