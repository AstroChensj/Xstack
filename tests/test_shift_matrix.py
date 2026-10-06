import numpy as np
import pytest
from astropy.io import fits

from Xstack.utils.rsp import (
    get_prob,
    get_tlmin_from_header,
    shift_matrix,
    shift_matrix_reference,
)


REDSHIFTS = (0.0, 0.001, 0.01, 0.03, 0.03474, 0.1, 0.201, 0.5, 1.0)


def _response(nrows, ncols):
    values = np.arange(1, nrows * ncols + 1, dtype=np.float64)
    response = values.reshape(nrows, ncols)
    response[response % 5 == 0] = 0.0
    return response


@pytest.mark.parametrize("z", REDSHIFTS)
@pytest.mark.parametrize(
    "iene_edges,ene_edges",
    [
        (
            np.linspace(1.0, 9.0, 9, dtype=np.float32),
            np.linspace(1.0, 9.0, 9, dtype=np.float32),
        ),
        (
            np.array([0.2, 0.3, 0.45, 0.7, 1.1, 1.8, 3.0], dtype=np.float32),
            np.array([0.1, 0.18, 0.31, 0.55, 0.9, 1.5, 2.4, 4.0], dtype=np.float32),
        ),
    ],
)
def test_optimized_shift_is_bitwise_equal_to_reference(z, iene_edges, ene_edges):
    iene_lo = iene_edges[:-1]
    iene_hi = iene_edges[1:]
    ene_lo = ene_edges[:-1]
    ene_hi = ene_edges[1:]
    response = _response(len(iene_lo), len(ene_lo))

    expected = shift_matrix_reference(
        response, iene_lo, iene_hi, ene_lo, ene_hi, z
    )
    actual = shift_matrix(response, iene_lo, iene_hi, ene_lo, ene_hi, z)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected, equal_nan=True)


def test_zero_redshift_returns_an_independent_copy():
    edges = np.linspace(1.0, 5.0, 5, dtype=np.float32)
    response = _response(4, 4)

    shifted = shift_matrix(
        response, edges[:-1], edges[1:], edges[:-1], edges[1:], 0.0
    )

    assert np.array_equal(shifted, response)
    assert shifted is not response


def test_response_shape_must_match_energy_grids():
    edges = np.linspace(1.0, 5.0, 5, dtype=np.float32)

    with pytest.raises(ValueError, match="does not match energy grids"):
        shift_matrix(
            np.zeros((3, 4)),
            edges[:-1],
            edges[1:],
            edges[:-1],
            edges[1:],
            0.1,
        )


def test_real_demo_response_is_bitwise_equal_to_reference():
    arf_path = "demo/data/sample.arf"
    rmf_path = "demo/data/sample.rmf"
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

    response = probability * specresp[:, np.newaxis]
    expected = shift_matrix_reference(
        response, iene_lo, iene_hi, ene_lo, ene_hi, 0.201
    )
    actual = shift_matrix(
        response, iene_lo, iene_hi, ene_lo, ene_hi, 0.201
    )

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected, equal_nan=True)
