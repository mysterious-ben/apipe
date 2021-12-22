# A-Pipe

**A-Pipe** allows to create data pipelines with lazy computation and caching.

**Features:**
- Lazy computation and cache loading
- Pickle and parquet serialization
- Support for hashing of `numpy` arrays and `pandas` DataFrames
- Support for `dask.Delayed` objects

## Installation

```shell
pip install apipe
```

## Examples

### Simple function caching

```python
import time
import apipe
import numpy as np
from loguru import logger

@apipe.eager_cached()
def load_data(table: str):
    time.sleep(1)
    arr = np.ones(5)
    logger.debug(f"transferred array data from table={table}")
    return arr

logger.info("start loading data")

# --- First pass: transfer data and save on disk
data = load_data("weather-ldn")
logger.info(f"finished loading data: {load_data()}")

# --- Second pass: load data from disk
data = load_data("weather-ldn")
logger.info(f"finished loading data: {load_data()}")
```


### Data pipeline with lazy execution and caching

```python
import apipe
import pandas as pd
import numpy as np
from loguru import logger

# --- Define data transformations via step functions (similar to dask.delayed)

@apipe.delayed_cached()  # lazy computation + caching on disk
def load_1():
    df = pd.DataFrame({"a": [1., 2.], "b": [0.1, np.nan]})
    logger.debug("Loaded {} records".format(len(df)))
    return df

@apipe.delayed_cached()  # lazy computation + caching on disk
def load_2(timestamp):
    df = pd.DataFrame({"a": [0.9, 3.], "b": [0.001, 1.]})
    logger.debug("Loaded {} records".format(len(df)))
    return df

@apipe.delayed_cached()  # lazy computation + caching on disk
def compute(x, y, eps):
    assert x.shape == y.shape
    diff = ((x - y).abs() / (y.abs()+eps)).mean().mean()
    logger.debug("Difference is computed")
    return diff

# --- Define pipeline dependencies
ts = pd.Timestamp(2019, 1, 1)
eps = 0.01
s1 = load_1()
s2 = load_2(ts)
diff = compute(s1, s2, eps)

# --- Trigger pipeline execution (first pass: compute everything and save on disk)
logger.info("diff: {:.3f}".format(apipe.delayed_compute((diff, ))[0]))

# --- Trigger pipeline execution (second pass: load from disk the end result only)
logger.info("diff: {:.3f}".format(apipe.delayed_compute((diff, ))[0]))
```

See more examples in a [notebook](https://github.com/mysterious-ben/ds-examples/blob/master/dataflows/dask_delayed_with_caching.ipynb).