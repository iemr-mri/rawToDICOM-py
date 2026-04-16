"""JCAMP-DX parameter file parser for Bruker ParaVision scan directories.

Translates readBrukerParamFile.m — parses acqp, method, visu_pars, and any
other Bruker JCAMP-DX text file into a plain Python dict.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def parse_bruker_params(path: Path) -> dict[str, Any]:
    """Parse a Bruker JCAMP-DX parameter file into a dict.

    Handles all value types:
      - Scalar numbers and strings
      - Numeric 1-D / 2-D arrays     (##$FIELD=( sizes ) \\n values)
      - String arrays                 (values wrapped in < >)
      - Struct arrays                 (parenthesised comma-separated groups)
      - Dynamic enums                 (##$FIELD=( <value>, <qualifier> ))
      - RLE-compressed arrays         (@N*(value))
    """
    with open(path, "r", encoding="latin-1") as fh:
        lines = fh.read().splitlines()

    params: dict[str, Any] = {}
    i = 0

    # Skip file header until first ##$ parameter line
    while i < len(lines) and not lines[i].startswith("##$"):
        i += 1

    while i < len(lines):
        line = lines[i]

        if not line or line.startswith("$$"):
            i += 1
            continue

        if line.startswith("##END"):
            break

        if not line.startswith("##$"):
            i += 1
            continue

        eq = line.index("=")
        field = line[3:eq]
        value_str = line[eq + 1 :]

        # Dynamic enum:  ( <GO_32BIT_SGN_INT>, <default> )
        if re.match(r"\(\s*<", value_str):
            data, i = _collect_continuation(lines, i, value_str)
            params[field] = _parse_dynamic_enum(data)

        # Array / struct with size declaration:  ( sizes... )
        elif value_str.startswith("("):
            data, i = _collect_continuation(lines, i, value_str)
            params[field] = _parse_array_value(data)

        else:
            # Scalar — inline comment after $$ is stripped
            raw = value_str.split("$$")[0].strip()
            params[field] = _parse_scalar(raw)
            i += 1

    return params


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _collect_continuation(lines: list[str], i: int, first: str) -> tuple[str, int]:
    """Collect the value_str and any continuation lines for an array parameter.

    Returns (combined_string, next_line_index).
    """
    parts = [first]
    j = i + 1
    while j < len(lines):
        nxt = lines[j]
        if not nxt or nxt.startswith("##") or nxt.startswith("$$"):
            break
        parts.append(nxt)
        j += 1
    return " ".join(parts), j


def _parse_scalar(raw: str) -> int | float | str:
    """Parse a scalar JCAMP value (number or bare string)."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_dynamic_enum(data: str) -> str:
    """Parse ( <value>, <qualifier> ) dynamic enum — return first <value>."""
    match = re.search(r"<([^>]*)>", data)
    return match.group(1) if match else data


def _expand_rle(data: str) -> str:
    """Expand @N*(val) RLE notation.  @4*(0) → 0 0 0 0."""

    def replacer(m: re.Match[str]) -> str:
        count = int(m.group(1))
        val = m.group(2)
        return " ".join([val] * count)

    return re.sub(r"@(\d+)\*\(([^)]+)\)", replacer, data)


def _parse_array_value(data: str) -> Any:
    """Parse an array declaration ( sizes ) followed by data.

    Dispatches to string-array, struct-array, or numeric/mixed-array parser.
    """
    sizes_match = re.match(r"\(\s*([\d\s,]+)\s*\)", data)
    if not sizes_match:
        return _parse_struct_array(data, [1])

    sizes = [int(s.strip()) for s in sizes_match.group(1).split(",") if s.strip()]
    rest = _expand_rle(data[sizes_match.end() :].strip())

    if not rest:
        return np.array([])

    if rest.startswith("<"):
        return _parse_string_array(rest)

    if rest.startswith("("):
        return _parse_struct_array(rest, sizes)

    return _parse_numeric_array(rest, sizes)


def _parse_string_array(data: str) -> str | list[str]:
    """Parse a sequence of <string> values.  Single string → plain str."""
    strings: list[str] = re.findall(r"<([^>]*)>", data)
    if len(strings) == 1:
        return strings[0]
    return strings


def _parse_numeric_array(data: str, sizes: list[int]) -> npt.NDArray[Any]:
    """Parse a whitespace-separated sequence of numbers (or mixed tokens).

    Numeric-only → reshaped numeric ndarray.
    Mixed / non-numeric → object ndarray reshaped to sizes.
    """
    tokens = data.split()
    values: list[Any] = []
    all_numeric = True
    for t in tokens:
        try:
            values.append(int(t))
        except ValueError:
            try:
                values.append(float(t))
            except ValueError:
                values.append(t)
                all_numeric = False

    if not values:
        return np.array([])

    if all_numeric:
        arr: npt.NDArray[Any] = np.array(values)
        total = int(np.prod(sizes))
        if len(arr) == total and len(sizes) > 1:
            return arr.reshape(sizes)
        return arr

    obj: npt.NDArray[Any] = np.empty(sizes if sizes else [len(values)], dtype=object)
    obj.flat[: len(values)] = values
    return obj


def _parse_struct_array(data: str, sizes: list[int]) -> npt.NDArray[Any]:
    """Parse one or more parenthesised struct groups into a 2-D object array.

    Returns shape (n_fields, prod(sizes)).  Access pattern:
        result[field_index, struct_index]
    matches MATLAB's cell matrix indexing.
    """
    groups = _extract_paren_groups(data)
    if not groups:
        return np.array([], dtype=object)

    parsed: list[list[Any]] = []
    for group in groups:
        fields = _split_by_comma(group)
        parsed.append([_parse_struct_field(f.strip()) for f in fields])

    n_fields = len(parsed[0])
    n_structs = len(parsed)
    result: npt.NDArray[Any] = np.empty((n_fields, n_structs), dtype=object)
    for j, struct_fields in enumerate(parsed):
        for idx, val in enumerate(struct_fields):
            result[idx, j] = val
    return result


def _extract_paren_groups(data: str) -> list[str]:
    """Return contents of top-level parenthesised groups."""
    groups = []
    depth = 0
    start = -1
    for idx, ch in enumerate(data):
        if ch == "(":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append(data[start:idx])
                start = -1
    return groups


def _split_by_comma(s: str) -> list[str]:
    """Split s by commas, ignoring commas inside < > or ( )."""
    parts: list[str] = []
    depth = 0
    in_str = False
    buf: list[str] = []
    for ch in s:
        if ch == "<":
            in_str = True
            buf.append(ch)
        elif ch == ">":
            in_str = False
            buf.append(ch)
        elif ch == "(" and not in_str:
            depth += 1
            buf.append(ch)
        elif ch == ")" and not in_str:
            depth -= 1
            buf.append(ch)
        elif ch == "," and not in_str and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _parse_struct_field(token: str) -> Any:
    """Parse one field value from a struct group (string, int, or float)."""
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token
