import datetime as dt
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest

from apipe._cached import _hash_obj


def test_hash_obj_short_str():
    obj = "ddddddd"
    hash_ = _hash_obj(obj)
    assert hash_ == obj


def test_hash_obj_short_int():
    obj = 23300
    hash_ = _hash_obj(obj)
    assert hash_ == str(obj)


def test_hash_obj_shortens_long_str():
    obj = "a" * 100
    hash_ = _hash_obj(obj, max_len=10)
    hash_ != obj


big_strings = (
    "a" * 10000,
    "a" * 9999,
    # "b" + "a" * 9999,
    "a" * 4999 + "b" + "a" * 5000,
    "a" * 5000 + "b" + "a" * 4999,
    # "a" * 9999 + "b",
)

big_int_arrays = (
    np.arange(10000),
    np.concatenate((np.asarray([8]), np.arange(1, 10000))),
    np.concatenate((np.arange(4999), np.arange(5000, 10000))),
    np.concatenate((np.arange(4999), np.asarray([8888]), np.arange(5000, 10000))),
    np.concatenate((np.arange(5000), np.asarray([8888]), np.arange(4999, 10000))),
)

big_datetime_arrays = tuple(
    np.asarray([dt.datetime(2010, 1, 1, 0, 0, 0) + dt.timedelta(minutes=int(x)) for x in arr])
    for arr in big_int_arrays
)

big_timedelta_arrays = tuple(
    np.asarray([dt.timedelta(minutes=int(x)) for x in arr]) for arr in big_int_arrays
)

big_datetime64_arrays = tuple(
    np.asarray(
        [np.datetime64("2010-01-01T00:00:00") + np.timedelta64(int(x), "m") for x in arr],
        dtype=np.datetime64,
    )
    for arr in big_int_arrays
)

pandas_series = (
    pd.Series(range(5)),
    pd.Series(range(1, 6)),
    pd.Series(range(5), index=range(1, 6)),
    pd.Series(range(5), name="series_a"),
    pd.Series(range(5), name="series_b"),
)

pandas_frames = (
    pd.DataFrame(
        {
            "a": range(5),
            "b": range(5),
        },
    ),
    pd.DataFrame(
        {
            "a": range(5),
            "b": range(1, 6),
        },
    ),
    pd.DataFrame(
        {
            "a": range(5),
            "b": range(5),
        },
        index=range(1, 6),
    ),
    pd.DataFrame(
        {
            "a": range(5),
            "c": range(5),
        },
    ),
)


@pytest.mark.parametrize(
    "objects",
    [
        big_strings,
        big_int_arrays,
        big_datetime_arrays,
        big_timedelta_arrays,
        big_datetime64_arrays,
        pandas_series,
        pandas_frames,
    ],
)
def test_hash_obj_discrimination(objects: Tuple[Any]):
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            hash_i = _hash_obj(objects[i])
            hash_j = _hash_obj(objects[j])
            assert hash_i != hash_j, f"{hash_i} ({i}) != {hash_j} ({j})"
