#!/usr/bin/env python3
"""Require bit-for-bit equality of two sets of Xstack FITS products."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits


FITS_SUFFIXES = {".fits", ".fit", ".pi", ".pha", ".arf", ".rmf", ".rsp"}
VOLATILE_HEADERS = {"DATE", "CHECKSUM", "DATASUM"}
FILE_LINK_HEADERS = {"ANCRFILE", "BACKFILE", "CORRFILE", "RESPFILE"}


def _fits_files(root):
    if root.is_file():
        return {root.name: root}
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in FITS_SUFFIXES
    }


def _equal_array(left, right):
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and np.array_equal(left_array, right_array, equal_nan=True)
    )


def _compare_column(left, right, label, differences):
    if len(left) != len(right):
        differences.append(f"{label}: row count {len(left)} != {len(right)}")
        return
    for row, (left_value, right_value) in enumerate(zip(left, right)):
        if not _equal_array(left_value, right_value):
            differences.append(f"{label}[{row}]: scientific values differ")
            return


def _header_value(keyword, value):
    if keyword in FILE_LINK_HEADERS and isinstance(value, str):
        return os.path.basename(value)
    return value


def _compare_header(left, right, label, differences):
    left_values = {}
    right_values = {}
    for header, destination in ((left, left_values), (right, right_values)):
        command_history = False
        for card in header.cards:
            keyword = card.keyword
            if keyword in VOLATILE_HEADERS or keyword == "":
                continue
            if keyword == "HISTORY":
                if str(card.value).startswith("CMD:"):
                    command_history = True
                if command_history:
                    continue
            destination.setdefault(keyword, []).append(
                _header_value(keyword, card.value)
            )
    if left_values != right_values:
        keys = sorted(set(left_values) | set(right_values))
        changed = [key for key in keys if left_values.get(key) != right_values.get(key)]
        differences.append(f"{label}: header values differ for {changed}")


def _compare_file(reference, candidate):
    differences = []
    with fits.open(reference, memmap=False) as left, fits.open(
        candidate, memmap=False
    ) as right:
        if len(left) != len(right):
            return [f"HDU count {len(left)} != {len(right)}"]
        for index, (left_hdu, right_hdu) in enumerate(zip(left, right)):
            label = f"HDU {index} ({left_hdu.name})"
            if left_hdu.name != right_hdu.name:
                differences.append(f"{label}: name != {right_hdu.name}")
            _compare_header(left_hdu.header, right_hdu.header, label, differences)

            left_data = left_hdu.data
            right_data = right_hdu.data
            if left_data is None or right_data is None:
                if left_data is not None or right_data is not None:
                    differences.append(f"{label}: only one HDU has data")
                continue

            left_names = getattr(left_data, "names", None)
            right_names = getattr(right_data, "names", None)
            if left_names is None or right_names is None:
                if not _equal_array(left_data, right_data):
                    differences.append(f"{label}: image data differ")
                continue
            if tuple(left_names) != tuple(right_names):
                differences.append(
                    f"{label}: columns {tuple(left_names)} != {tuple(right_names)}"
                )
                continue
            for name in left_names:
                _compare_column(
                    left_data[name],
                    right_data[name],
                    f"{label}/{name}",
                    differences,
                )
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    reference = _fits_files(args.reference)
    candidate = _fits_files(args.candidate)
    report = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "missing_from_candidate": sorted(set(reference) - set(candidate)),
        "extra_in_candidate": sorted(set(candidate) - set(reference)),
        "files": {},
    }
    for relative_path in sorted(set(reference) & set(candidate)):
        differences = _compare_file(reference[relative_path], candidate[relative_path])
        report["files"][relative_path] = {
            "equal": not differences,
            "differences": differences,
        }

    report["equal"] = (
        not report["missing_from_candidate"]
        and not report["extra_in_candidate"]
        and all(item["equal"] for item in report["files"].values())
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.write_text(serialized + "\n")
    print(serialized)
    raise SystemExit(0 if report["equal"] else 1)


if __name__ == "__main__":
    main()
