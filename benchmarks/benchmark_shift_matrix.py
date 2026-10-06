#!/usr/bin/env python3
"""Benchmark one Xstack response-matrix shift and emit a JSON result."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Xstack.utils.rsp import (
    get_prob,
    get_tlmin_from_header,
    shift_matrix,
    shift_matrix_reference,
)


def _load_response(arf_path, rmf_path):
    with fits.open(arf_path, memmap=False) as hdus:
        arf = hdus["SPECRESP"].data
        iene_lo = arf["ENERG_LO"].astype(np.float32)
        iene_hi = arf["ENERG_HI"].astype(np.float32)
        specresp = np.asarray(arf["SPECRESP"])

    with fits.open(rmf_path, memmap=False) as hdus:
        matrix = hdus["MATRIX"].data
        ebounds = hdus["EBOUNDS"].data
        ene_lo = ebounds["E_MIN"].astype(np.float32)
        ene_hi = ebounds["E_MAX"].astype(np.float32)
        probability = get_prob(
            matrix,
            ebounds,
            get_tlmin_from_header(rmf_path),
        )

    return (
        probability * specresp[:, np.newaxis],
        iene_lo,
        iene_hi,
        ene_lo,
        ene_hi,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arf", required=True, type=Path)
    parser.add_argument("--rmf", required=True, type=Path)
    parser.add_argument("--redshift", required=True, type=float)
    parser.add_argument(
        "--implementation",
        required=True,
        choices=("reference", "optimized"),
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Compile and run once before the measured call.",
    )
    args = parser.parse_args()

    response = _load_response(args.arf, args.rmf)
    function = (
        shift_matrix_reference
        if args.implementation == "reference"
        else shift_matrix
    )
    if args.warmup:
        function(*response, args.redshift)

    started = time.perf_counter()
    shifted = function(*response, args.redshift)
    elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)

    print(
        json.dumps(
            {
                "implementation": args.implementation,
                "warmup": args.warmup,
                "redshift": args.redshift,
                "shape": list(shifted.shape),
                "dtype": str(shifted.dtype),
                "elapsed_seconds": elapsed,
                "peak_rss_kib": usage.ru_maxrss,
                "arf": str(args.arf.resolve()),
                "rmf": str(args.rmf.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
