from abc import ABC
from typing import Dict, Union, Type


_COUNTERS: Dict[str, int] = {}


def get_default_name(key: Union[str, Type, object]) -> str:
    if isinstance(key, str):
        # TODO: This will need a lock to support multithreading.
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
    name: str

    def __init__(self, *, name: str = None):
        self.name = name or get_default_name(self)

    def __str__(self) -> str:
        return self.name
