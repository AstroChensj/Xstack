import numpy as np
import pytest
from astropy.io import fits

from Xstack.utils.rsp import get_prob, get_tlmin_from_header


def _write_rmf(path, tlmin):
    nene = 2
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=np.arange(4)),
            fits.Column(
                name="E_MIN",
                format="E",
                array=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            fits.Column(
                name="E_MAX",
                format="E",
                array=np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            ),
        ],
        name="EBOUNDS",
    )
    matrix = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="ENERG_LO",
                format="E",
                array=np.array([1.0, 2.0], dtype=np.float32),
            ),
            fits.Column(
                name="ENERG_HI",
                format="E",
                array=np.array([2.0, 3.0], dtype=np.float32),
            ),
            fits.Column(name="N_GRP", format="I", array=np.ones(nene, dtype=np.int16)),
            fits.Column(
                name="F_CHAN",
                format="PI()",
                array=[np.array([0], dtype=np.int16) for _ in range(nene)],
            ),
            fits.Column(
                name="N_CHAN",
                format="PI()",
                array=[np.array([2], dtype=np.int16) for _ in range(nene)],
            ),
            fits.Column(
                name="MATRIX",
                format="PE()",
                array=[np.array([0.75, 0.25], dtype=np.float32) for _ in range(nene)],
            ),
        ],
        name="MATRIX",
    )
    if tlmin is not None:
        matrix.header["TLMIN4"] = tlmin
    fits.HDUList([fits.PrimaryHDU(), ebounds, matrix]).writeto(path)


@pytest.mark.parametrize("header_value", [0, "0"])
def test_get_tlmin_returns_int_for_numeric_header_values(tmp_path, header_value):
    rmf = tmp_path / "response.rmf"
    _write_rmf(rmf, header_value)

    value = get_tlmin_from_header(rmf)

    assert value == 0
    assert type(value) is int


def test_get_tlmin_defaults_to_one_when_keyword_is_missing(tmp_path):
    rmf = tmp_path / "response.rmf"
    _write_rmf(rmf, None)

    value = get_tlmin_from_header(rmf)

    assert value == 1
    assert type(value) is int


def test_string_tlmin_produces_same_probability_matrix_as_integer(tmp_path):
    integer_rmf = tmp_path / "integer.rmf"
    string_rmf = tmp_path / "string.rmf"
    _write_rmf(integer_rmf, 0)
    _write_rmf(string_rmf, "0")

    probabilities = []
    for rmf in (integer_rmf, string_rmf):
        with fits.open(rmf) as hdus:
            probabilities.append(
                get_prob(
                    hdus["MATRIX"].data,
                    hdus["EBOUNDS"].data,
                    get_tlmin_from_header(rmf),
                )
            )

    np.testing.assert_array_equal(probabilities[0], probabilities[1])
