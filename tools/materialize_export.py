# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export committed CoC sources with self-contained, verified RSG files."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess


def materialize(destination):
    """Copy committed package files and verified RSG sources to a new directory."""
    project = Path(__file__).resolve().parents[1]
    checkout = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=project, text=True
        ).strip()
    )
    prefix = str(project.relative_to(checkout))
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", prefix], cwd=checkout, check=True)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", prefix], cwd=checkout
    )
    if untracked:
        raise ValueError("commit package files before materializing an export")
    spec = importlib.util.spec_from_file_location(
        "_coc_export_build", project / "src/_coc_build_support.py"
    )
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    package, selected = support.verify_rsg_source(project)
    source_prefix = str(package.resolve().relative_to(checkout))
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", source_prefix], cwd=checkout, check=True
    )
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "--",
            *[source_prefix + "/" + name for name in selected],
        ],
        cwd=checkout,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    names = (
        subprocess.check_output(
            ["git", "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", prefix], cwd=checkout
        )
        .decode()
        .split("\0")
    )
    destination = Path(destination)
    if destination.exists():
        raise ValueError("export destination must not exist")
    destination.mkdir(parents=True)
    for name in filter(None, names):
        source = checkout / name
        relative = source.relative_to(project)
        if str(relative) == "src/rsg_labeling":
            continue
        if source.is_symlink():
            raise ValueError(f"unapproved export symlink: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in selected:
        target = destination / "src/rsg_labeling" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    support.verify_rsg_source(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(materialize(parser.parse_args().destination))
