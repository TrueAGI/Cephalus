"""This module provides common utility and support functions for the rest of the package."""

from typing import Union, Sequence, Tuple

import tensorflow as tf

__all__ = [
    'RawTensorShape',
    'StandardizedTensorShape',
    'standardize_tensor_shape',
    'size_from_shape',
]

RawTensorShape = Union[int, Sequence[int], tf.Tensor]
StandardizedTensorShape = Tuple[int, ...]


def standardize_tensor_shape(tensor_shape: RawTensorShape) -> StandardizedTensorShape:
    """Convert a tensor shape to a standardized representation: a tuple of ints."""
    try:
        return int(tensor_shape),
    except ValueError:
        return tuple(int(dim) for dim in tensor_shape)


def size_from_shape(tensor_shape: RawTensorShape) -> int:
    """Given a tensor's shape, return the tensor's size."""
    return int(tf.reduce_prod(standardize_tensor_shape(tensor_shape)))
