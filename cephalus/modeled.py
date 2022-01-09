from abc import abstractmethod
from typing import Tuple

import tensorflow as tf

from cephalus.names import Named


class Modeled(Named):
    _built: bool = False

    @abstractmethod
    def build(self) -> None:
        self._built = True

    @abstractmethod
    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        """Return a tuple of the trainable weights of the neural models used by the modeled
        object."""
        return ()

    @property
    def is_built(self) -> bool:
        return self._built
