# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.data import group_by_window
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops


# TODO: replace with official TF version
def bucket_by_sequence_length(element_length_func,
                              bucket_boundaries,
                              bucket_batch_sizes,
                              padded_shapes=None,
                              padding_values=None,
                              pad_to_bucket_boundary=False):
    """A transformation that buckets elements in a `Dataset` by length.
    Elements of the `Dataset` are grouped together by length and then are padded
    and batched.
    This is useful for sequence tasks in which the elements have variable length.
    Grouping together elements that have similar lengths reduces the total
    fraction of padding in a batch which increases training step efficiency.
    Args:
      element_length_func: function from element in `Dataset` to `tf.int64`,
        determines the length of the element, which will determine the bucket it
        goes into.
      bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
      bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
        `len(bucket_boundaries) + 1`.
      padded_shapes: Nested structure of `tf.TensorShape` to pass to
        @{tf.data.Dataset.padded_batch}. If not provided, will use
        `dataset.output_shapes`, which will result in variable length dimensions
        being padded out to the maximum length in each batch.
      padding_values: Values to pad with, passed to
        @{tf.data.Dataset.padded_batch}. Defaults to padding with 0.
      pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
        size to maximum length in batch. If `True`, will pad dimensions with
        unknown size to bucket boundary, and caller must ensure that the source
        `Dataset` does not contain any elements with length longer than
        `max(bucket_boundaries)`.
    Returns:
      A `Dataset` transformation function, which can be passed to
      @{tf.data.Dataset.apply}.
    Raises:
      ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
    """
    with ops.name_scope("bucket_by_seq_length"):
        if len(bucket_batch_sizes) != (len(bucket_boundaries) + 1):
            raise ValueError(
                "len(bucket_batch_sizes) must equal len(bucket_boundaries) + 1")

        batch_sizes = constant_op.constant(bucket_batch_sizes, dtype=dtypes.int64)

        def element_to_bucket_id(element):
            """Return int64 id of the length bucket for this element."""
            seq_length = element_length_func(element)

            boundaries = list(bucket_boundaries)
            buckets_min = [np.iinfo(np.int32).min] + boundaries
            buckets_max = boundaries + [np.iinfo(np.int32).max]
            conditions_c = math_ops.logical_and(
                math_ops.less_equal(buckets_min, seq_length),
                math_ops.less(seq_length, buckets_max))
            bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))

            return bucket_id

        def window_size_fn(bucket_id):
            # The window size is set to the batch size for this bucket
            window_size = batch_sizes[bucket_id]
            return window_size

        def make_padded_shapes(shapes, none_filler=None):
            padded = []
            for shape in nest.flatten(shapes):
                shape = tensor_shape.TensorShape(shape)
                shape = [
                    none_filler if d.value is None else d
                    for d in shape
                ]
                padded.append(shape)
            return nest.pack_sequence_as(shapes, padded)

        def batching_fn(bucket_id, grouped_dataset):
            """Batch elements in dataset."""
            batch_size = batch_sizes[bucket_id]
            none_filler = None
            if pad_to_bucket_boundary:
                err_msg = ("When pad_to_bucket_boundary=True, elements must have "
                           "length <= max(bucket_boundaries).")
                check = check_ops.assert_less(
                    bucket_id,
                    constant_op.constant(len(bucket_batch_sizes) - 1,
                                         dtype=dtypes.int64),
                    message=err_msg)
                with ops.control_dependencies([check]):
                    boundaries = constant_op.constant(bucket_boundaries,
                                                      dtype=dtypes.int64)
                    bucket_boundary = boundaries[bucket_id]
                    none_filler = bucket_boundary
            shapes = make_padded_shapes(
                padded_shapes or grouped_dataset.output_shapes,
                none_filler=none_filler)
            return grouped_dataset.padded_batch(batch_size, shapes, padding_values)

        def _apply_fn(dataset):
            return dataset.apply(
                group_by_window(element_to_bucket_id, batching_fn,
                                window_size_func=window_size_fn))

        return _apply_fn
