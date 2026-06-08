import numpy as np

from insurance_ml.config import INTERACTION_COLS, TARGET
from insurance_ml.features import add_interaction_features, base_feature_list, build_feature_matrix
from insurance_ml.preprocessing import preprocess


def _featurized(df_raw):
    df_encoded, region_cols = preprocess(df_raw)
    return add_interaction_features(df_encoded, df_raw), region_cols


def test_interaction_features_match_hand_computation(df_raw):
    df_feat, _ = _featurized(df_raw)

    np.testing.assert_allclose(df_feat["smoker_bmi"], df_feat["smoker_enc"] * df_feat["bmi"])
    np.testing.assert_allclose(df_feat["age_sq"], df_feat["age"] ** 2)
    np.testing.assert_allclose(df_feat["smoker_obese"], df_feat["smoker_enc"] * df_feat["bmi_obese"])
    assert set(df_feat["bmi_obese"].unique()) <= {0, 1}
    assert (df_feat.loc[df_raw["bmi"] >= 30, "bmi_obese"] == 1).all()
    for col in INTERACTION_COLS:
        assert col in df_feat.columns


def test_base_feature_list_includes_region_dummies():
    feats = base_feature_list(["region_northeast", "region_northwest"])
    assert feats == ["age", "bmi", "children", "sex_enc", "smoker_enc",
                     "region_northeast", "region_northwest"]


def test_feature_selection_returns_consistent_subsets(df_raw):
    df_feat, region_cols = _featurized(df_raw)
    selection = build_feature_matrix(df_feat, region_cols, config_seed=42)

    candidate_pool = set(selection["all_feats"])
    assert set(selection["final_feats"]) <= candidate_pool
    assert set(selection["selected_rfe"]) <= candidate_pool
    assert TARGET not in selection["final_feats"]
    # core domain features must always survive the union rule
    for must_have in ("age", "bmi", "smoker_enc", "smoker_bmi", "smoker_age"):
        assert must_have in selection["final_feats"]
