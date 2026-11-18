#!/usr/bin/env python3
"""Cross-platform structural checks for the LIMO ROS repository."""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "limo_control"
IGNORED_DIRS = {".git", "build", "devel", "install", "logs", "__pycache__"}
BAD_SUFFIXES = {".bak", ".pyc", ".pyo"}
REQUIRED_PATHS = {
    ROOT / "README.md",
    ROOT / "LICENSE",
    PACKAGE / "CMakeLists.txt",
    PACKAGE / "package.xml",
    PACKAGE / "global_planner_plugin.xml",
    PACKAGE / "scripts" / "limo_patrol.py",
    PACKAGE / "scripts" / "lidar_avoidance_node.py",
    PACKAGE / "scripts" / "depth_avoidance_node.py",
    PACKAGE / "src" / "my_global_planner.cpp",
    PACKAGE / "src" / "limo_control" / "patrol_modules" / "__init__.py",
}


def repository_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def main() -> int:
    errors: list[str] = []
    files = list(repository_files())

    missing = sorted(path for path in REQUIRED_PATHS if not path.is_file())
    errors.extend(f"missing required path: {path.relative_to(ROOT)}" for path in missing)

    package_manifests = sorted(ROOT.rglob("package.xml"))
    package_manifests = [p for p in package_manifests if ".git" not in p.parts]
    if package_manifests != [PACKAGE / "package.xml"]:
        rendered = ", ".join(str(p.relative_to(ROOT)) for p in package_manifests)
        errors.append(f"expected one ROS package manifest, found: {rendered}")

    for path in files:
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in BAD_SUFFIXES or path.name.endswith("_backup.launch"):
            errors.append(f"generated or backup artifact is tracked: {rel}")

        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python parse failed for {rel}: {exc}")

        if path.suffix in {".xml", ".launch"} or path.name == "package.xml":
            try:
                ET.parse(path)
            except (ET.ParseError, OSError) as exc:
                errors.append(f"XML parse failed for {rel}: {exc}")

    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        if path.stat().st_size == 0:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(path.relative_to(ROOT))

    for duplicates in hashes.values():
        if len(duplicates) > 1:
            rendered = ", ".join(str(path) for path in sorted(duplicates))
            errors.append(f"exact duplicate files: {rendered}")

    if errors:
        print("Repository checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Repository checks passed: {len(files)} files, one ROS package, no exact duplicates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
