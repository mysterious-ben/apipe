import functools
import json
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import xxhash
from loguru import logger

xxhasher = xxhash.xxh64(seed=42)

MAX_ARG_HASH_LEN = 32  # limit length of hash string
MAX_NAME_LEN = 240  # limit length of cache file name (not counting file extention)
CACHE_FOLDER = "cache"


def _hash_ndarray(arr: np.ndarray) -> str:
    try:
        data = arr if np.issubdtype(arr.dtype, np.number) else cloudpickle.dumps(arr)
    except TypeError:  # numpy cannot determine data type
        data = cloudpickle.dumps(arr)
    try:
        xxhasher.update(data)
    except ValueError:  # cannot hash M-type array
        data = cloudpickle.dumps(arr)
        xxhasher.update(data)
    h = str(xxhasher.intdigest())
    return h


def _hash_obj(obj, max_len: Optional[int] = MAX_ARG_HASH_LEN) -> str:
    xxhasher.reset()
    if isinstance(obj, np.ndarray):
        h = _hash_ndarray(obj)
    elif isinstance(obj, pd.Series):
        h = str(obj.name)
        h += _hash_ndarray(obj.index.values)
        h += _hash_ndarray(obj.values)
    elif isinstance(obj, pd.DataFrame):
        h_ = _hash_ndarray(obj.index.values)
        for c in obj:
            h_ += str(c) + _hash_ndarray(obj[c].values)
        xxhasher.update(h_)
        h = str(xxhasher.intdigest())
    elif (
        isinstance(obj, str)
        or isinstance(obj, bool)
        or isinstance(obj, int)
        or isinstance(obj, float)
        or isinstance(obj, complex)
    ):
        h = str(obj)
    elif isinstance(obj, bytes):
        xxhasher.update(obj)
        h = str(xxhasher.intdigest())
    else:
        pickled_obj = cloudpickle.dumps(obj)
        xxhasher.update(pickled_obj)
        h = str(xxhasher.intdigest())
    if (max_len is not None) and (len(h) > max_len):
        xxhasher.update(h)
        h_sffx = str(xxhasher.intdigest())
        h = f"{h[: max_len - len(h_sffx) - 1]}-{h_sffx}"
    return h


def _hash_obj_cached(obj, max_len: int = MAX_ARG_HASH_LEN) -> str:
    if hasattr(obj, "__cached_hash__"):
        h = obj.__cached_hash__()
    else:
        h = _hash_obj(obj, max_len)
    return h


def _kw_is_private(k: str) -> bool:
    """Return true if this keyword argument should be excluded from cache filename"""
    return k.startswith("_")


def _get_cache_name(
    name: Optional[str],
    name_prefix: str,
    parameters: Optional[dict],
    ignore_args: Optional[bool],
    ignore_kwargs: Optional[Union[bool, List[str]]],
    kwargs_sep: str,
    foo: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    max_name_len: Optional[int] = MAX_NAME_LEN,
) -> str:
    if ignore_args is None:
        ignore_args = parameters is not None
    if ignore_kwargs is None:
        ignore_kwargs = parameters is not None
    if name is None:
        name = foo.__name__
    _n = [name]
    if parameters is not None:
        _n.extend(
            [
                str(k) + kwargs_sep + _hash_obj_cached(v)
                for k, v in parameters.items()
                if not _kw_is_private(k)
            ]
        )
    if not ignore_args:
        _n.extend([_hash_obj_cached(a) for a in args])
    if not ignore_kwargs:
        _n.extend(
            [
                str(k) + kwargs_sep + _hash_obj_cached(v)
                for k, v in kwargs.items()
                if not _kw_is_private(k)
            ]
        )
    elif isinstance(ignore_kwargs, list) or isinstance(ignore_kwargs, set):
        _n.extend(
            [
                str(k) + kwargs_sep + _hash_obj_cached(v)
                for k, v in kwargs.items()
                if k not in ignore_kwargs
            ]
        )
    else:
        assert isinstance(ignore_kwargs, bool)
    full_name = name_prefix + "_".join(_n)
    if (max_name_len is not None) and (len(full_name) > max_name_len):
        xxhasher.reset()
        xxhasher.update(full_name)
        h_sffx = str(xxhasher.intdigest())
        full_name = f"{full_name[: max_name_len - len(h_sffx) - 1]}-{h_sffx}"
    return full_name


def _cached_load(ftype: str, path: Path) -> Any:
    if ftype == "parquet":
        data = pd.read_parquet(path)
    elif ftype == "pickle":
        data = cloudpickle.load(open(path, "rb"))
    else:
        raise ValueError("ftype {} is not recognized".format(ftype))
    return data


def _cached_save(data: Any, ftype: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ftype == "parquet":
        data.to_parquet(path, index=True, allow_truncated_timestamps=True)
    elif ftype == "pickle":
        cloudpickle.dump(data, open(path, "wb"))
    else:
        raise ValueError("ftype {} is not recognized".format(ftype))


class CacheMeta:
    """Cache meta data (incl. hash)"""

    def __init__(
        self,
        name: str,
        folder: Union[Path, str],
        ftype: str,
        nout: Optional[int],
        hash_value: Optional[str],
    ):
        self.name = name
        self._folder = Path(folder).absolute()
        self.ftype = ftype
        self.nout = nout
        self.hash_value = hash_value

    @staticmethod
    def _get_meta_path(folder: Union[Path, str], name: str) -> Path:
        return Path(folder).absolute() / (name + ".meta")

    @property
    def meta_path(self) -> Path:
        return self._get_meta_path(folder=self._folder, name=self.name)

    @property
    def cache_path(self) -> Union[Path, List[Path]]:
        if self.nout is None:
            return self._folder / (self.name + f".{self.ftype}")
        else:
            return [self._folder / (self.name + f"__{i}.{self.ftype}") for i in range(self.nout)]

    @classmethod
    def from_file(cls, folder: Union[Path, str], name: str) -> "CacheMeta":
        """Load metadata from cache metafile"""

        meta_path = cls._get_meta_path(folder=folder, name=name)
        if meta_path.exists():
            with open(meta_path, "rt") as f:
                fields = json.load(f)
            return cls(folder=folder, **fields)
        else:
            raise FileNotFoundError(f"Meta file is missing: {meta_path}")

    def dump_to_file(self) -> None:
        """Dump metadata to cache metafile"""

        with open(self.meta_path, "wt") as f:
            fields = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            json.dump(fields, f)


class CachedResult:
    """Lazy loader for cache data"""

    def __init__(self, meta: CacheMeta, verbose: bool = True):
        self.meta = meta
        self.verbose = verbose
        self._cache_value: Optional[Any] = None
        self._lock_dump_load = multiprocessing.Lock()
        self._lock_hash = multiprocessing.Lock()

    def load(self) -> Any:
        """Load data from cache"""

        if self._cache_value is None:
            with self._lock_dump_load:
                _cache_path = self.meta.cache_path
                if isinstance(_cache_path, Path):
                    self._cache_value = _cached_load(self.meta.ftype, _cache_path)
                elif isinstance(_cache_path, List):
                    self._cache_value = tuple(
                        _cached_load(self.meta.ftype, cp) for cp in _cache_path
                    )
                else:
                    raise ValueError(f"Unexpected meta.cache_path={_cache_path}")
            if self.verbose:
                logger.debug("Task {}: data has been loaded from cache".format(self.meta.name))
        return self._cache_value

    def dump(self, data):
        """Update and dump data to cache"""

        with self._lock_dump_load:
            self._cache_value = data
            if self.meta.nout is None:
                _cached_save(self._cache_value, self.meta.ftype, self.meta.cache_path)
            else:
                assert len(self._cache_value) == self.meta.nout
                assert len(self.meta.cache_path) == self.meta.nout
                for cv, cp in zip(self._cache_value, self.meta.cache_path):
                    _cached_save(cv, self.meta.ftype, cp)
        if self.verbose:
            logger.debug("Task {}: data has been saved to cache".format(self.meta.name))
        self.meta.dump_to_file()

    def __cached_hash__(self):
        """Get hash of cached data

        Used to construct a cache file name
        """

        # Hash may not be required, so it's not automatically computed from data
        if self.meta.hash_value is None:
            with self._lock_hash:
                cache_obj = self.load()  # activates _lock_dump_load
                if self.meta.nout is None:
                    self.meta.hash_value = _hash_obj(cache_obj)
                else:
                    self.meta.hash_value = tuple(_hash_obj(co) for co in cache_obj)
                self.meta.dump_to_file()
            if self.verbose:
                logger.debug("Task {}: hash has been computed from data".format(self.meta.name))
        return self.meta.hash_value

    def exists(self):
        return self.meta.meta_path.exists()


class CachedResultItem:
    """Lazy loader for cache data

    A wrapper for CachedResult class to deal with functions that return multiple outputs
    """

    def __init__(self, result: CachedResult, item: Any):
        self.result = result
        self.item = item

    def load(self) -> Any:
        if self.item is None:
            return self.result.load()
        else:
            return self.result.load()[self.item]

    def dump(self, data) -> None:
        self.result.dump(data)

    def __cached_hash__(self) -> str:
        if self.item is None:
            return self.result.__cached_hash__()
        else:
            return self.result.__cached_hash__()[self.item]


def load(
    cached_result: Union[CachedResultItem, Iterable[CachedResultItem]]
) -> Union[Any, Tuple[Any, ...]]:
    """Load cached objects"""

    if isinstance(cached_result, CachedResultItem):
        return cached_result.load()
    else:
        return tuple(r.load() for r in cached_result)


def cached(
    name: Optional[str] = None,
    name_prefix: str = "",
    parameters: Optional[dict] = None,
    ignore_args: Optional[bool] = None,
    ignore_kwargs: Optional[Union[bool, List[str]]] = None,
    folder: Union[str, Path] = CACHE_FOLDER,
    ftype: str = "pickle",
    kwargs_sep: str = "|",
    nout: Optional[int] = None,
    override: bool = False,
    verbose: bool = True,
):
    """Cache function output on the disk

    Features:
    - Lazy computation and cache loading
    - Hashing of complex arguments (arrays and DataFrames)
    - Pickle and parquet serialization

    Args:
        name: name of the cache file
            if none, name is constructed from the function name and args
        parameters: include these parameters in the name
            only meaningful when `name=None`
            Improtant: parameters starting with _ (underscore) will be ignored
        ignore_args: if true, do not add args to the name
        ignore_kwargs: if true, do not add kwargs to the name
            it's also possible to specify a list of kwargs to ignore
            Improtant: kwargs starting with _ (underscore) will be ignored
        folder: name of the cache folder
        ftype: type of the cache file
            'pickle' | 'parquet'
        kwargs_sep: string separating a keyword parameter and its value
        override: if true, override the existing cache file
        verbose: if true, log caching and loading information

    Returns: function with caching
        `output = foo()` will lazily load data from cache if exists, otherwise lazily compute
        `data = output.load()` will eager load/compute data
    """

    def decorator(foo):
        """Cache function output on the disk"""

        @functools.wraps(foo)
        def cached_foo(*args, **kwargs) -> Union[CachedResultItem, Tuple[CachedResultItem, ...]]:
            fullname = _get_cache_name(
                name=name,
                name_prefix=name_prefix,
                parameters=parameters,
                ignore_args=ignore_args,
                ignore_kwargs=ignore_kwargs,
                kwargs_sep=kwargs_sep,
                foo=foo,
                args=args,
                kwargs=kwargs,
            )
            try:
                meta = CacheMeta.from_file(folder=folder, name=fullname)
            except FileNotFoundError as e:
                if verbose:
                    logger.debug(f"Cache not found: {str(e)}")
                meta = CacheMeta(
                    name=fullname, folder=folder, ftype=ftype, nout=nout, hash_value=None
                )
            result = CachedResult(meta=meta, verbose=verbose)
            output: Union[CachedResultItem, Tuple[CachedResultItem, ...]]
            if not override and result.exists():
                # if the result (= cache OR cache + hash) exists, do nothing - just pass it on
                # the cache will be loaded only if required later
                if nout is not None:
                    output = tuple(CachedResultItem(result, i) for i in range(nout))
                else:
                    output = CachedResultItem(result, None)
                if verbose:
                    logger.info("Task {}: skip (cache exists)".format(result.meta.name))
            else:
                # if the result does not exist, generate data and save cache
                # eager load all CachedResultItem arguments
                args = tuple(a.load() if isinstance(a, CachedResultItem) else a for a in args)
                kwargs = {
                    k: v.load() if isinstance(v, CachedResultItem) else v
                    for k, v in kwargs.items()
                }
                data = foo(*args, **kwargs)
                result.dump(data)
                if nout is not None:
                    output = tuple(CachedResultItem(result, i) for i in range(nout))
                else:
                    output = CachedResultItem(result, None)
                if verbose:
                    logger.info(
                        "Task {}: data has been computed and saved to cache".format(
                            result.meta.name
                        )
                    )
            return output

        return cached_foo

    return decorator


def eager_cached(
    name: Optional[str] = None,
    name_prefix: str = "",
    parameters: Optional[dict] = None,
    ignore_args: Optional[bool] = None,
    ignore_kwargs: Optional[Union[bool, List[str]]] = None,
    folder: Union[str, Path] = CACHE_FOLDER,
    ftype: str = "pickle",
    kwargs_sep: str = "|",
    nout: Optional[int] = None,
    override: bool = False,
    verbose: bool = True,
):
    """Cache function output on the disk (eager load)"""

    def decorator(foo):
        """Cache function output on the disk (eager load)"""

        cached_foo = cached(
            name=name,
            name_prefix=name_prefix,
            parameters=parameters,
            ignore_args=ignore_args,
            ignore_kwargs=ignore_kwargs,
            folder=folder,
            ftype=ftype,
            kwargs_sep=kwargs_sep,
            nout=nout,
            override=override,
            verbose=verbose,
        )(foo)

        @functools.wraps(foo)
        def eager_cached_foo(*args, **kwargs):
            cached_output = cached_foo(*args, **kwargs)
            return load(cached_output)

        return eager_cached_foo

    return decorator


def clear_cache(
    folder: Union[str, Path] = CACHE_FOLDER,
    ignore_errors: bool = True,
):
    """Clear the cache folder

    Args:
        folder: name of the cache folder
    """
    folder = Path(folder).absolute()
    shutil.rmtree(folder, ignore_errors=ignore_errors)
