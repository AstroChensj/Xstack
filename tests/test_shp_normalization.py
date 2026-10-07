import numpy as np

from Xstack.utils.rsp import (
    extract_arf_rmf_from_rspmat,
    get_folded_model_rate,
    rescale_rspmat,
)


ENE_LO = np.array([1.0, 2.0, 3.0])
ENE_HI = np.array([2.0, 3.0, 4.0])
IENE_LO = np.array([1.0, 2.0])
IENE_HI = np.array([2.0, 4.0])
FLAG = np.array([True, True, False])

SHP_RAW = np.array([
    [2.0, 1.0, 0.5],
    [0.25, 3.0, 1.0],
])
FLX_RAW = np.array([
    [5.0, 0.5, 1.0],
    [0.75, 2.0, 0.25],
])
LMN_RAW = FLX_RAW * 1e-55

SHP_WEIGHTS = np.array([2.0, 3.0])
EXPOSURES = np.array([10.0, 30.0])
REGAREAS = np.ones(2)


def _rescale_shp(normalization, norm_rspmat=None):
    return rescale_rspmat(
        rspmat=SHP_RAW.copy(),
        rspwt_lst=SHP_WEIGHTS.copy(),
        expo_lst=EXPOSURES,
        rega_lst=REGAREAS,
        rspwt_method="SHP",
        shp_normalization=normalization,
        norm_rspmat=None if norm_rspmat is None else norm_rspmat.copy(),
        ene_lo=ENE_LO,
        ene_hi=ENE_HI,
        iene_lo=IENE_LO,
        iene_hi=IENE_HI,
        flg=FLAG,
        gamma=2.0,
    )


def test_get_folded_model_rate_matches_direct_calculation():
    model = ((IENE_LO + IENE_HI) / 2.0) ** -2.0
    widths = IENE_HI - IENE_LO
    folded = np.sum(
        SHP_RAW * model[:, np.newaxis] * widths[:, np.newaxis],
        axis=0,
    )
    expected = np.sum(folded[FLAG])

    actual = get_folded_model_rate(
        SHP_RAW, ENE_LO, ENE_HI, IENE_LO, IENE_HI, FLAG, gamma=2.0,
    )
    assert actual == expected


def test_legacy_shp_is_bitwise_identical_to_old_formula():
    expected_rsp = SHP_RAW.copy()
    expected_weights = SHP_WEIGHTS.copy()
    expected_norm = 1.0 / np.sum(expected_weights)
    expected_rsp *= expected_norm
    expected_weights *= expected_norm

    rsp, rspnorm, weights, expo, rega, shp_renorm, norm_rate = _rescale_shp("LEGACY")

    assert np.array_equal(rsp, expected_rsp)
    assert np.array_equal(weights, expected_weights)
    assert rspnorm == expected_norm
    assert expo == np.sum(EXPOSURES)
    assert rega == 1.0
    assert shp_renorm == 1.0
    assert norm_rate is None


def test_shp_flx_preserves_shape_and_matches_flx_folded_rate():
    legacy_rsp, *_ = _rescale_shp("LEGACY")
    shp_flx_rsp, _, _, expo, rega, renorm, _ = _rescale_shp("FLX", FLX_RAW)
    flx_rsp, _, _, flx_expo, flx_rega, _, _ = rescale_rspmat(
        FLX_RAW.copy(), np.array([10.0, 30.0]), EXPOSURES, REGAREAS, "FLX",
    )

    shp_rate = get_folded_model_rate(
        shp_flx_rsp, ENE_LO, ENE_HI, IENE_LO, IENE_HI, FLAG,
    )
    flx_rate = get_folded_model_rate(
        flx_rsp, ENE_LO, ENE_HI, IENE_LO, IENE_HI, FLAG,
    )
    assert np.isclose(shp_rate, flx_rate, rtol=1e-14, atol=0.0)
    assert np.allclose(shp_flx_rsp, legacy_rsp * renorm, rtol=0.0, atol=0.0)
    assert expo == flx_expo
    assert rega == flx_rega

    _, legacy_rmf = extract_arf_rmf_from_rspmat(legacy_rsp)
    _, physical_rmf = extract_arf_rmf_from_rspmat(shp_flx_rsp)
    assert np.allclose(physical_rmf, legacy_rmf, rtol=1e-15, atol=0.0)


def test_shp_lmn_matches_lmn_rate_and_keeps_1e60_convention():
    shp_lmn_rsp, _, _, expo, rega, renorm, _ = _rescale_shp("LMN", LMN_RAW)
    lmn_rsp, rspnorm, _, lmn_expo, lmn_rega, _, _ = rescale_rspmat(
        LMN_RAW.copy(), np.array([1e-55, 2e-55]),
        EXPOSURES, REGAREAS, "LMN",
    )

    shp_rate = get_folded_model_rate(
        shp_lmn_rsp, ENE_LO, ENE_HI, IENE_LO, IENE_HI, FLAG,
    )
    lmn_rate = get_folded_model_rate(
        lmn_rsp, ENE_LO, ENE_HI, IENE_LO, IENE_HI, FLAG,
    )
    assert np.isclose(shp_rate, lmn_rate, rtol=1e-14, atol=0.0)
    assert rspnorm == 1e60
    assert expo == lmn_expo
    assert rega == lmn_rega
    assert np.isfinite(renorm) and renorm > 0.0


def test_shp_normalization_is_ignored_for_ordinary_flx():
    result = rescale_rspmat(
        FLX_RAW.copy(), np.array([10.0, 30.0]),
        EXPOSURES, REGAREAS, "FLX", shp_normalization="LMN",
    )
    expected = FLX_RAW.copy()
    expected *= 1.0 / np.sum(EXPOSURES)
    assert np.array_equal(result[0], expected)
    assert result[-2] == 1.0
    assert result[-1] is None
