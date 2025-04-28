import pandas as pd
import numpy as np
import pytest

from match_forecast.utils.functions import (
    clean_and_impute,
    agg_positions,
    merge_and_select_average,
    make_diff_features
)
from match_forecast.config import POSITION_MAP


def test_make_diff_features():
    # Create a simple DataFrame with HOME_ and AWAY_ cols
    df = pd.DataFrame({
        'HOME_val': [1, 2, 3],
        'AWAY_val': [3, 2, 1],
        'other': [10, 20, 30]
    }, index=[0, 1, 2])
    out = make_diff_features(df.copy())
    # DIFF_val should exist
    assert 'DIFF_val' in out.columns
    # Originals dropped
    assert 'HOME_val' not in out.columns
    assert 'AWAY_val' not in out.columns
    # Diff correctness
    expected = pd.Series([-2, 0, 2], index=[0,1,2])
    pd.testing.assert_series_equal(out['DIFF_val'], expected)
    # Other column preserved
    assert 'other' in out.columns


def test_clean_and_impute_numeric_and_categorical(tmp_path):
    # Numeric with NaN and categorical with NaN
    df = pd.DataFrame({
        'ID': [1,2,3],
        'num': [1.0, np.nan, 3.0],
        'cat': ['a', None, 'b']
    }).set_index('ID')
    out = clean_and_impute(df.copy(), home=True, meta_cols=None, threshold=0.5)
    # Columns prefixed
    assert 'HOME_num' in out.columns
    assert 'HOME_cat' in out.columns
    # No missing remain
    assert not out['HOME_num'].isna().any()
    assert not out['HOME_cat'].isna().any()


def test_agg_positions_grouped_mean():
    # Setup DataFrame with two IDs and positions
    df = pd.DataFrame({
        'ID': [1,1,2,2],
        'POSITION': ['attacker','midfielder','defender','goalkeeper'],
        'feat_average': [10,20,30,40]
    }).set_index('ID')
    out = agg_positions(df, mapping=POSITION_MAP, id_col='ID', pos_col='POSITION')
    # Offensive = mean of attacker and midfielder = 15
    assert np.isclose(out.loc[1, 'feat_average_offensive'], 15.0)
    # Defender and goalkeeper direct
    assert np.isclose(out.loc[2, 'feat_average_defender'], 30.0)
    assert np.isclose(out.loc[2, 'feat_average_goalkeeper'], 40.0)


def test_merge_and_select_average_filters_suffix():
    home = pd.DataFrame({'ID':[1,2], 'a_average':[1,2], 'b_other':[5,6]})
    away = pd.DataFrame({'ID':[1,2], 'a_average':[2,1], 'c_average':[7,8]})
    out = merge_and_select_average(home, away, id_col='ID', suffix='_average')
    # Should keep a_average from both, c_average, drop b_other
    assert list(out.columns) == ['a_average', 'c_average']
    # Values match merge
    assert out.loc[1,'a_average'] == 1 and out.loc[2,'a_average'] == 2


if __name__ == '__main__':
    pytest.main()
