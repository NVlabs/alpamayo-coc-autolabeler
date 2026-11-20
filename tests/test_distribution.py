# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise real artifacts and the canonical-source release boundary."""

import email
import importlib.util
import ast
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("setuptools", minversion="77")


@pytest.fixture
def distribution_source(tmp_path):
    """Create a distribution fixture from the package sources."""
    project = Path(__file__).resolve().parents[1]
    destination = tmp_path / "project"
    shutil.copytree(
        project,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"
        ),
    )
    alias = destination / "src/rsg_labeling"
    if alias.is_symlink():
        alias.unlink()
        shutil.copytree(
            (project / "src/rsg_labeling").resolve(),
            alias,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    return destination


def _build(project, kind):
    """Run the actual PEP 517 backend without network or model execution."""
    output = project / "dist"
    output.mkdir(exist_ok=True)
    command = f"import setuptools.build_meta as b; b.build_{kind}({str(output)!r})"
    result = subprocess.run(
        [sys.executable, "-c", command], cwd=project, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return next(output.glob("*.whl" if kind in {"wheel", "editable"} else "*.tar.gz"))


def test_distribution_contains_self_contained_rsg_sources(distribution_source, tmp_path):
    """Build wheel/sdist and rebuild from the archive with identical selected code."""
    project = distribution_source
    package = project / "src/rsg_labeling"
    manifest = json.loads((package / "releases/files.json").read_text())
    paths = set(manifest["files"])
    wheel = _build(project, "wheel")
    with zipfile.ZipFile(wheel) as archive:
        actual = {
            p.removeprefix("rsg_labeling/")
            for p in archive.namelist()
            if p.startswith("rsg_labeling/") and not p.endswith("/")
        }
        assert actual == paths
        for path in paths:
            assert archive.read("rsg_labeling/" + path) == (package / path).read_bytes()
        metadata = email.message_from_bytes(
            archive.read(next(p for p in archive.namelist() if p.endswith(".dist-info/METADATA")))
        )
        assert metadata["Requires-Python"] == ">=3.11"
        base = {
            x.split(";")[0].split("<")[0].split(">=")[0]
            for x in metadata.get_all("Requires-Dist")
            if ";" not in x
        }
        assert base == {"numpy", "pandas", "pydantic"}
        build_info = ast.parse(archive.read("coc_labeling/_build_info.py"))
        identity = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in build_info.body
            if isinstance(node, ast.Assign)
        }
        assert identity["VERSION"] == "2.0.0"
        assert len(identity["SOURCE_SHA256"]) == 64
        expected_build_info = archive.read("coc_labeling/_build_info.py")

    sdist = _build(project, "sdist")
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(sdist) as archive:
        members = [
            m for m in archive.getmembers() if "/src/rsg_labeling/" in m.name and not m.isdir()
        ]
        assert len(members) == len(paths)
        assert all(m.isfile() for m in members)
        archive.extractall(extracted, filter="data")
    standalone = next(extracted.iterdir())
    assert not (standalone / "src/rsg_labeling").is_symlink()
    rebuilt = _build(standalone, "wheel")
    with zipfile.ZipFile(rebuilt) as archive:
        assert archive.read("coc_labeling/_build_info.py") == expected_build_info
        for path in paths:
            assert archive.read("rsg_labeling/" + path) == (package / path).read_bytes()


@pytest.mark.parametrize("change", ["extra", "missing", "symlink"])
def test_distribution_rejects_unselected_missing_or_linked_source(distribution_source, change):
    """Reject undeclared files, missing source and links before building artifacts."""
    project = distribution_source
    spec = importlib.util.spec_from_file_location(
        "_test_build_support", project / "src/_coc_build_support.py"
    )
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    if change == "extra":
        (project / "src/rsg_labeling/private_unselected.py").write_text("private = True\n")
    elif change == "symlink":
        (project / "src/rsg_labeling/unselected_directory").symlink_to(
            project.parent, target_is_directory=True
        )
    else:
        (project / "src/rsg_labeling/common/coc.py").unlink()
    with pytest.raises(RuntimeError, match="RSG source"):
        support.verify_rsg_source(project)


def test_source_edits_change_component_identity(distribution_source):
    """Allow edits to selected code while identifying the changed build."""
    project = distribution_source
    spec = importlib.util.spec_from_file_location(
        "_test_edit_build_support", project / "src/_coc_build_support.py"
    )
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    before = support.component_build_info(project)
    path = project / "src/rsg_labeling/__init__.py"
    path.write_bytes(path.read_bytes() + b"\n# Consumer source edit\n")
    support.verify_rsg_source(project)
    after = support.component_build_info(project)
    assert before[0] == after[0]
    assert before[1] != after[1]


def test_editable_install_preserves_source_checkout_identity(distribution_source):
    """Keep the existing editable standalone install without modifying source identity."""
    project = distribution_source
    source_identity = project / "src/coc_labeling/_build_info.py"
    before = source_identity.read_bytes()
    wheel = _build(project, "editable")
    with zipfile.ZipFile(wheel) as archive:
        assert any(name.endswith(".pth") for name in archive.namelist())
    assert source_identity.read_bytes() == before
