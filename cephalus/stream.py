"""A harness which links a state kernel to its environment, coordinating their interactions."""

from typing import Callable, Optional, TypeVar, Generic

from cephalus.frame import StateFrame
from cephalus.kernel import StateKernel

__all__ = [
    'StateStream'
]

from cephalus.names import Named

Environment = TypeVar('Environment')


class StateStream(Named, Generic[Environment]):
    """Harnesses a state kernel to make a sequence of state predictions within an environment."""

    def __init__(self, kernel: StateKernel[Environment], environment: Environment = None, *,
                 name: str = None):
        self._kernel = kernel
        self._environment = environment
        self._previous_frame: Optional[StateFrame] = None

        super().__init__(name=name)

    @property
    def kernel(self) -> StateKernel[Environment]:
        return self._kernel

    @property
    def environment(self) -> Environment:
        return self._environment

    def run(self, steps: int = None, terminate: Callable[[], bool] = None) -> None:
        """Run the kernel in the environment. If steps is provided, run for at most that many
        steps. If terminate callback is provided, call it just before each step and stop if it
        returns True."""
        step = 0
        while True:
            if steps is not None and step >= steps:
                break
            if terminate is not None and terminate():
                break
            step += 1
            self.step()

    def step(self) -> None:
        """Run the kernel for a single step in the environment."""
        frame = self._kernel.step(self._environment, self._previous_frame, self.name)
        self._previous_frame = frame
