"""Named objects."""

from abc import ABC
from typing import Dict, Union, Type
from threading import Lock


_COUNTER_LOCK = Lock()
_COUNTERS: Dict[str, int] = {}


def get_default_name(key: Union[str, Type, object]) -> str:
    """Return a decent default name for objects whose creators didn't bother to provide one."""

    if isinstance(key, str):
        with _COUNTER_LOCK:
            counter = _COUNTERS.get(key, 0)
            _COUNTERS[key] = counter + 1
        return '%s-%s' % (key, counter)
    elif hasattr(key, '__qualname__') or hasattr(key, '__name__'):
        module_name = getattr(key, '__module__', '')
        key_name = getattr(key, '__qualname__', getattr(key, '__name__'))
        key = module_name + ':' + key_name
        return get_default_name(key)
    else:
        return get_default_name(type(key))


class Named(ABC):
    """Abstract base class for named objects."""

    name: str

    def __init__(self, *, name: str = None):
        self.name = name or get_default_name(self)

    def __str__(self) -> str:
        return self.name
