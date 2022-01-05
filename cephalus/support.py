"""This module provides common utility and support functions for the rest of the package."""

from typing import Union, Sequence, Tuple

import tensorflow as tf

__all__ = [
    'RawTensorShape',
    'StandardizedTensorShape',
    'standardize_tensor_shape',
    'size_from_shape',
    'OnlineStats'
]

RawTensorShape = Union[int, Sequence[int], tf.Tensor]
StandardizedTensorShape = Tuple[int, ...]


def standardize_tensor_shape(tensor_shape: RawTensorShape) -> StandardizedTensorShape:
    """Convert a tensor shape to a standardized representation: a tuple of ints."""
    try:
        return int(tensor_shape),
    except (TypeError, ValueError):
        return tuple(int(dim) for dim in tensor_shape)


def size_from_shape(tensor_shape: RawTensorShape) -> int:
    """Given a tensor's shape, return the tensor's size."""
    return int(tf.reduce_prod(standardize_tensor_shape(tensor_shape)))


class OnlineStats:
    """Compute the mean, variance, and stddev of a sequence of values, as the values are
    received."""

    def __init__(self):
        self.mean = 0.0
        self.sq_mean = 0.0
        self.measurements = 0

    def update(self, sample: float) -> None:
        self.measurements += 1
        self.mean += (sample - self.mean) / self.measurements
        self.sq_mean += (sample * sample - self.sq_mean) / self.measurements

    @property
    def variance(self) -> float:
        return abs(self.sq_mean - self.mean * self.mean)

    @property
    def stddev(self) -> float:
        return self.variance ** 0.5
