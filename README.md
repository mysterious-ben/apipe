# A-Pipe

**A-Pipe** allows to create data pipelines with lazy computation and caching.

**Features:**
- Lazy computation and cache loading
- Pickle and parquet serialization
- Support for hashing of `numpy` arrays and `pandas` DataFrames
- Support of Delayed objects

## Installation

```shell
pip install apipe
```

## Example

```python
import apipe
import pandas as pd
import numpy as np
from loguru import logger

# --- Define data transformations via step functions (similar to dask.delayed)

@apipe.delayed_cached()  # lazy computation + caching on disk
def load_1():
    df = pd.DataFrame({'a': [1., 2.], 'b': [0.1, np.nan]})
    logger.info('Loaded {} records'.format(len(df)))
    return df

@apipe.delayed_cached()  # lazy computation + caching on disk
def load_2(timestamp):
    df = pd.DataFrame({'a': [0.9, 3.], 'b': [0.001, 1.]})
    logger.info('Loaded {} records'.format(len(df)))
    return df

@apipe.delayed_cached()  # lazy computation + caching on disk
def compute(x, y, eps):
    assert x.shape == y.shape
    diff = ((x - y).abs() / (y.abs()+eps)).mean().mean()
    logger.info('Difference is computed')
    return diff

# --- Define pipeline dependencies
ts = pd.Timestamp(2019, 1, 1)
eps = 0.01
s1 = load_1()
s2 = load_2(ts)
diff = compute(s1, s2, eps)

# --- Trigger pipeline execution
print('diff: {:.3f}'.format(apipe.delayed_compute((diff, ))[0]))
```
