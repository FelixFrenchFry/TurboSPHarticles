#!/usr/bin/env python3
"""Generate deterministic geometry metadata for assets/models/rigid_models.json."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = pathlib.Path(__file__).resolve().parent / "rigid_models.json"
DEFAULT_MODEL_DIR = pathlib.Path(__file__).resolve().parent / "rigid_meshes"


class MetadataError(RuntimeError):
    pass


@dataclass
class EdgeInfo:
    count: int
    orientation: int


def _parse_face_index(token: str, vertex_count: int, path: pathlib.Path, line_number: int) -> int:
    index_text = token.split("/", 1)[0]
    if not index_text:
        raise MetadataError(f"{path}:{line_number}: empty face vertex index")
    try:
        obj_index = int(index_text)
    except ValueError as error:
        raise MetadataError(f"{path}:{line_number}: invalid face vertex index {index_text!r}") from error
    if obj_index == 0:
        raise MetadataError(f"{path}:{line_number}: OBJ vertex indices cannot be zero")
    index = obj_index - 1 if obj_index > 0 else vertex_count + obj_index
    if index < 0 or index >= vertex_count:
        raise MetadataError(f"{path}:{line_number}: face vertex index {obj_index} is out of range")
    return index


def analyze_obj(path: pathlib.Path) -> dict[str, object]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except OSError as error:
        raise MetadataError(f"Could not read model {path}: {error}") from error

    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise MetadataError(f"{path}:{line_number}: vertex requires three coordinates")
            try:
                vertex = (float(fields[1]), float(fields[2]), float(fields[3]))
            except ValueError as error:
                raise MetadataError(f"{path}:{line_number}: invalid vertex coordinate") from error
            if not all(math.isfinite(value) for value in vertex):
                raise MetadataError(f"{path}:{line_number}: vertex coordinates must be finite")
            vertices.append(vertex)
        elif fields[0] == "f":
            face_tokens = []
            for token in fields[1:]:
                if token.startswith("#"):
                    break
                face_tokens.append(token)
            if len(face_tokens) < 3:
                raise MetadataError(f"{path}:{line_number}: face requires at least three vertices")
            face = [
                _parse_face_index(token, len(vertices), path, line_number)
                for token in face_tokens
            ]
            for index in range(1, len(face) - 1):
                triangles.append((face[0], face[index], face[index + 1]))

    if not vertices or not triangles:
        raise MetadataError(f"{path}: model must contain vertices and triangle faces")

    minimum = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    extent = [maximum[axis] - minimum[axis] for axis in range(3)]

    edges: dict[tuple[int, int], EdgeInfo] = {}
    signed_volume = 0.0
    degenerate_triangle_count = 0

    for i0, i1, i2 in triangles:
        a = vertices[i0]
        b = vertices[i1]
        c = vertices[i2]
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        double_area = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
        repeated_index = i0 == i1 or i0 == i2 or i1 == i2
        if repeated_index or double_area == 0.0:
            degenerate_triangle_count += 1
        signed_volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0

        for start, end in ((i0, i1), (i1, i2), (i2, i0)):
            key = (start, end) if start < end else (end, start)
            orientation = 1 if start < end else -1
            edge = edges.get(key)
            if edge is None:
                edges[key] = EdgeInfo(1, orientation)
            else:
                edge.count += 1
                edge.orientation += orientation

    boundary_edge_count = sum(edge.count == 1 for edge in edges.values())
    non_manifold_edge_count = sum(edge.count > 2 for edge in edges.values())
    inconsistent_edge_count = sum(
        edge.count == 2 and edge.orientation != 0 for edge in edges.values()
    )
    watertight = boundary_edge_count == 0 and non_manifold_edge_count == 0
    enclosed_volume = (
        abs(signed_volume)
        if watertight and inconsistent_edge_count == 0 and degenerate_triangle_count == 0
        else None
    )

    return {
        "vertexCount": len(vertices),
        "triangleCount": len(triangles),
        "sourceExtent": extent,
        "enclosedVolume": enclosed_volume,
        "watertight": watertight,
    }


def generate_catalog(catalog_path: pathlib.Path, model_dir: pathlib.Path) -> tuple[dict, list[str]]:
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetadataError(f"Could not read catalog {catalog_path}: {error}") from error
    models = catalog.get("models")
    if not isinstance(models, dict):
        raise MetadataError(f"{catalog_path}: root 'models' value must be an object")

    changed: list[str] = []
    for model_name, model in models.items():
        if not isinstance(model, dict) or not isinstance(model.get("file"), str):
            raise MetadataError(f"{catalog_path}: model {model_name!r} requires a string 'file'")
        model_path = model_dir / model["file"]
        metadata = analyze_obj(model_path)
        if model.get("metadata") != metadata:
            changed.append(model_name)
        model["metadata"] = metadata
    return catalog, changed


def serialize_catalog(catalog: dict) -> str:
    text = json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False)
    number = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    vector_pattern = re.compile(
        rf"\[\n\s*({number}),\n\s*({number}),\n\s*({number})\n\s*\]"
    )
    return vector_pattern.sub(r"[\1, \2, \3]", text) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=pathlib.Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--check", action="store_true", help="fail if generated metadata differs")
    args = parser.parse_args(argv)

    try:
        catalog, changed = generate_catalog(args.catalog, args.model_dir)
        generated_text = serialize_catalog(catalog)
        existing_text = args.catalog.read_text(encoding="utf-8")
        if args.check:
            if generated_text != existing_text:
                names = ", ".join(changed) if changed else "JSON formatting"
                print(f"Model catalog is stale: {names}", file=sys.stderr)
                return 1
            print(f"Model catalog metadata is current for {len(catalog['models'])} models.")
            return 0

        args.catalog.write_text(generated_text, encoding="utf-8", newline="\n")
        print(f"Updated metadata for {len(catalog['models'])} models in {args.catalog}.")
        if changed:
            print("Changed entries: " + ", ".join(changed))
        return 0
    except MetadataError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
