import numpy as np

from insurance_ml.config import LOG_TARGET, NUMERIC_COLS, TARGET
from insurance_ml.preprocessing import add_log_target, apply_scaler, encode_categoricals, fit_scaler, preprocess


def test_encode_categoricals_drops_raw_strings_and_adds_binaries(df_raw):
    df, region_cols = encode_categoricals(df_raw)

    assert "sex" not in df.columns
    assert "smoker" not in df.columns
    assert set(df["sex_enc"].unique()) <= {0, 1}
    assert set(df["smoker_enc"].unique()) <= {0, 1}
    assert (df["sex_enc"] == (df_raw["sex"] == "male").astype(int)).all()
    assert region_cols and all(c.startswith("region_") for c in region_cols)
    assert df[region_cols].sum(axis=1).eq(1).all()  # exactly one region flag per row


def test_add_log_target_is_log1p(df_raw):
    df, _ = encode_categoricals(df_raw)
    df = add_log_target(df)

    assert LOG_TARGET in df.columns
    np.testing.assert_allclose(df[LOG_TARGET].values, np.log1p(df[TARGET].values))


def test_scaler_round_trips_to_zero_mean_unit_variance(df_raw):
    df, _ = preprocess(df_raw)
    scaler = fit_scaler(df)
    scaled = apply_scaler(df, scaler)

    means = scaled[NUMERIC_COLS].mean().values
    stds = scaled[NUMERIC_COLS].std(ddof=0).values
    np.testing.assert_allclose(means, np.zeros(len(NUMERIC_COLS)), atol=1e-8)
    np.testing.assert_allclose(stds, np.ones(len(NUMERIC_COLS)), atol=1e-8)
