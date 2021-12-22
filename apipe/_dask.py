# import functools
import multiprocessing
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import dask
from dask.delayed import Delayed

from apipe._cached import CACHE_FOLDER, CachedResultItem, _kw_is_private, cached


class DelayedParameter:
    """Delayed parameter = a Delayed object that can change the returned value

    Important! Methods `update` and `context` do not work with dask distributed.

    Args:
        name: parameter name
        value: parameter value
    """

    def __init__(self, name, value=None):
        self._name = name
        self._value = value
        self._delayed = dask.delayed(name=name)(lambda: self._value)()  # pylint: disable=E1120
        self._lock_context = multiprocessing.Lock()

    def set(self, value) -> None:
        """Permanently change the value of this parameter"""
        self._value = value

    def __call__(self) -> Delayed:
        """Get a Delayed object"""
        return self._delayed

    @contextmanager
    def context(self, value):
        """Change the value of this parameter within a context"""
        with self._lock_context:
            old_value = self._value
            self.set(value)
            yield
            self.set(old_value)


class DelayedParameters:
    """A dictionary of delayed parameters

    Important! Methods `update` and `context` do not work with dask distributed.
    """

    def __init__(self):
        self._params = {}
        self._param_delayed = {}
        self._lock_context = multiprocessing.Lock()

    def create(self, name: str, value: Any = None) -> Delayed:
        """Create a new parameter and return a delayed object"""
        if name in self._params:
            raise KeyError(f"Parameter {name} already exists")
        self._params[name] = value
        self._param_delayed[name] = dask.delayed(name=name)(  # pylint: disable=E1120
            lambda: self._params[name]
        )()
        return self._param_delayed[name]

    def create_many(self, d: dict) -> None:
        """Create multiple parameters at once"""
        for k, v in d.items():
            self.create(k, v)

    def get_params(self) -> dict:
        """Get parameters as a dictionary (name -> value)"""
        return {k: v for k, v in self._params.items() if not _kw_is_private(k)}

    def get_delayed(self, name: str) -> Delayed:
        """Get a Delayed object for the chosen parameter"""
        return self._param_delayed[name]

    def update(self, name: str, value: Any) -> None:
        """Permanently update parameter value"""
        if name not in self._params:
            raise KeyError(f"Parameter {name} does not exist")
        self._params[name] = value

    def update_many(self, d: dict) -> None:
        """Permanently update multiple parameter values"""
        for k, v in d.items():
            self.update(k, v)

    @contextmanager
    def context(self, d: dict):
        """Update multiple parameter values within a context"""
        with self._lock_context:
            old_params = dict(**self._params)
            self.update_many(d)
            yield
            self.update_many(old_params)


def _n_delayed_arguments(*args, **kwargs) -> int:
    n_delayed_args = sum(isinstance(a, Delayed) for a in args)
    n_delayed_kwargs = sum(isinstance(v, Delayed) for _, v in kwargs.items())
    return n_delayed_args + n_delayed_kwargs


def delayed_cached(
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
    """Lazy function execution with caching of the output on the disk

    dask.delayed + apipe.cached

    Args:
        see apipe.cached
    """

    def decorator(foo):
        """Delay and cache function output on disk (dask.delayed + apipe.cached)"""

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

        # @functools.wraps(foo)  # Can't use: Delayed objects are immutable
        @dask.delayed(name=name, pure=False, nout=nout)  # pylint: disable=E1120
        def delayed_cached_fool(*args, **kwargs):
            if _n_delayed_arguments(*args, **kwargs) > 0:
                return foo(*args, **kwargs)
            else:
                return cached_foo(*args, **kwargs)

        return delayed_cached_fool

    return decorator


def delayed_compute(tasks, scheduler="threads") -> Tuple:
    """Compute values of Delayed objects or load it from cache"""
    results = dask.compute(*tasks, scheduler=scheduler)
    datas = tuple(r.load() if isinstance(r, CachedResultItem) else r for r in results)
    return datas
