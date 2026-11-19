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

"""Check public source boundaries and generate reproducible component identity."""

import hashlib
import json
import tomllib
from pathlib import Path, PurePosixPath

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def verify_rsg_source(project_root=None):
    """Return the package directory and its declared relative files, rejecting extras/links."""
    project = Path(project_root) if project_root is not None else Path(__file__).parent.parent
    package = project / "src/rsg_labeling"
    manifest = json.loads((package / "releases/files.json").read_text())
    selected = []
    for name in manifest["files"]:
        path = PurePosixPath(name)
        if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != name:
            raise RuntimeError("invalid RSG source path")
        selected.append(name)
    if len(set(selected)) != len(selected):
        raise RuntimeError("duplicate RSG source path")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise RuntimeError("RSG source must not contain nested symlinks")
    actual = {
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if actual != set(selected):
        raise RuntimeError("RSG source differs from its declared file list")
    return package, tuple(sorted(selected))


def component_build_info(project):
    """Hash the code being packaged without embedding repository paths or review IDs."""
    sources = []
    for package in ("coc_labeling", "meta_action", "rsg_labeling"):
        for path in sorted((project / "src" / package).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.name == "_build_info.py":
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            sources.append(
                {
                    "path": str(path.relative_to(project)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    sources.append(
        {
            "path": "pyproject.toml",
            "sha256": hashlib.sha256((project / "pyproject.toml").read_bytes()).hexdigest(),
        }
    )
    digest = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    version = tomllib.loads((project / "pyproject.toml").read_text())["project"]["version"]
    return version, digest


class CheckedBuildPy(build_py):
    """Build modules after checking the declared public source boundaries."""

    def run(self):
        """Check sources and write an identity for the actual packaged code."""
        verify_rsg_source()
        super().run()
        version, digest = component_build_info(Path(__file__).parent.parent)
        path = Path(self.build_lib) / "coc_labeling/_build_info.py"
        path.write_text(
            '"""Deterministic component identity generated during distribution build."""\n'
            f"VERSION = {version!r}\nSOURCE_SHA256 = {digest!r}\n"
        )


class CheckedSdist(sdist):
    """Build an archive containing the declared public sources."""

    def run(self):
        """Check file selection before constructing the source archive."""
        verify_rsg_source()
        super().run()
