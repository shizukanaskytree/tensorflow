"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: stateful_random_ops.cc
"""

import collections as _collections
import six as _six

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import errors as _errors
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util.tf_export import kwarg_only as _kwarg_only
from tensorflow.tools.docs import doc_controls as _doc_controls


def non_deterministic_ints(shape, dtype=_dtypes.int64, name=None):
  r"""Non-deterministically generates some integers.

  This op may use some OS-provided source of non-determinism (e.g. an RNG), so each execution will give different results.

  Args:
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.int64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "NonDeterministicInts", name, _ctx._post_execution_callbacks, shape,
        "dtype", dtype)
      return _result
    except _core._FallbackException:
      try:
        return non_deterministic_ints_eager_fallback(
            shape, dtype=dtype, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "NonDeterministicInts", shape=shape, dtype=dtype, name=name)
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("dtype", _op.get_attr("dtype"), "shape_dtype",
            _op.get_attr("shape_dtype"))
  _execute.record_gradient(
      "NonDeterministicInts", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

@_doc_controls.do_not_generate_docs
@_kwarg_only
def NonDeterministicInts(shape, dtype):
  return non_deterministic_ints(shape=shape, dtype=dtype)
tf_export("raw_ops.NonDeterministicInts")(NonDeterministicInts)


def non_deterministic_ints_eager_fallback(shape, dtype=_dtypes.int64, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function non_deterministic_ints
  """
  _ctx = ctx if ctx else _context.context()
  if dtype is None:
    dtype = _dtypes.int64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int64)
  _inputs_flat = [shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"NonDeterministicInts", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "NonDeterministicInts", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def stateful_standard_normal(resource, shape, dtype=_dtypes.float32, name=None):
  r"""Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "StatefulStandardNormal", name, _ctx._post_execution_callbacks,
        resource, shape, "dtype", dtype)
      return _result
    except _core._FallbackException:
      try:
        return stateful_standard_normal_eager_fallback(
            resource, shape, dtype=dtype, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatefulStandardNormal", resource=resource, shape=shape, dtype=dtype,
                                  name=name)
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("dtype", _op.get_attr("dtype"), "shape_dtype",
            _op.get_attr("shape_dtype"))
  _execute.record_gradient(
      "StatefulStandardNormal", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

@_doc_controls.do_not_generate_docs
@_kwarg_only
def StatefulStandardNormal(resource, shape, dtype):
  return stateful_standard_normal(resource=resource, shape=shape, dtype=dtype)
tf_export("raw_ops.StatefulStandardNormal")(StatefulStandardNormal)


def stateful_standard_normal_eager_fallback(resource, shape, dtype=_dtypes.float32, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function stateful_standard_normal
  """
  _ctx = ctx if ctx else _context.context()
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulStandardNormal", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                             name=name)
  _execute.record_gradient(
      "StatefulStandardNormal", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def stateful_standard_normal_v2(resource, algorithm, shape, dtype=_dtypes.float32, name=None):
  r"""Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "StatefulStandardNormalV2", name, _ctx._post_execution_callbacks,
        resource, algorithm, shape, "dtype", dtype)
      return _result
    except _core._FallbackException:
      try:
        return stateful_standard_normal_v2_eager_fallback(
            resource, algorithm, shape, dtype=dtype, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatefulStandardNormalV2", resource=resource, algorithm=algorithm,
                                    shape=shape, dtype=dtype, name=name)
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("dtype", _op.get_attr("dtype"), "shape_dtype",
            _op.get_attr("shape_dtype"))
  _execute.record_gradient(
      "StatefulStandardNormalV2", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

@_doc_controls.do_not_generate_docs
@_kwarg_only
def StatefulStandardNormalV2(resource, algorithm, shape, dtype):
  return stateful_standard_normal_v2(resource=resource, algorithm=algorithm, shape=shape, dtype=dtype)
tf_export("raw_ops.StatefulStandardNormalV2")(StatefulStandardNormalV2)


def stateful_standard_normal_v2_eager_fallback(resource, algorithm, shape, dtype=_dtypes.float32, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function stateful_standard_normal_v2
  """
  _ctx = ctx if ctx else _context.context()
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulStandardNormalV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                             name=name)
  _execute.record_gradient(
      "StatefulStandardNormalV2", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def stateful_uniform_full_int(resource, algorithm, shape, dtype=_dtypes.uint64, name=None):
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers covering the whole range of `dtype`.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.uint64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "StatefulUniformFullInt", name, _ctx._post_execution_callbacks,
        resource, algorithm, shape, "dtype", dtype)
      return _result
    except _core._FallbackException:
      try:
        return stateful_uniform_full_int_eager_fallback(
            resource, algorithm, shape, dtype=dtype, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatefulUniformFullInt", resource=resource, algorithm=algorithm,
                                  shape=shape, dtype=dtype, name=name)
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("dtype", _op.get_attr("dtype"), "shape_dtype",
            _op.get_attr("shape_dtype"))
  _execute.record_gradient(
      "StatefulUniformFullInt", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

@_doc_controls.do_not_generate_docs
@_kwarg_only
def StatefulUniformFullInt(resource, algorithm, shape, dtype):
  return stateful_uniform_full_int(resource=resource, algorithm=algorithm, shape=shape, dtype=dtype)
tf_export("raw_ops.StatefulUniformFullInt")(StatefulUniformFullInt)


def stateful_uniform_full_int_eager_fallback(resource, algorithm, shape, dtype=_dtypes.uint64, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function stateful_uniform_full_int
  """
  _ctx = ctx if ctx else _context.context()
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape]
  _attrs = ("dtype", dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulUniformFullInt", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                             name=name)
  _execute.record_gradient(
      "StatefulUniformFullInt", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def stateful_uniform_int(resource, algorithm, shape, minval, maxval, name=None):
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers in the range `[minval, maxval)`.
  The lower bound `minval` is included in the range, while the upper bound
  `maxval` is excluded.

  The random integers are slightly biased unless `maxval - minval` is an exact
  power of two.  The bias is small for values of `maxval - minval` significantly
  smaller than the range of the output (either `2^32` or `2^64`).

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    minval: A `Tensor`. Minimum value (inclusive, scalar).
    maxval: A `Tensor`. Must have the same type as `minval`.
      Maximum value (exclusive, scalar).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "StatefulUniformInt", name, _ctx._post_execution_callbacks, resource,
        algorithm, shape, minval, maxval)
      return _result
    except _core._FallbackException:
      try:
        return stateful_uniform_int_eager_fallback(
            resource, algorithm, shape, minval, maxval, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  _, _, _op = _op_def_lib._apply_op_helper(
        "StatefulUniformInt", resource=resource, algorithm=algorithm,
                              shape=shape, minval=minval, maxval=maxval,
                              name=name)
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("dtype", _op.get_attr("dtype"), "shape_dtype",
            _op.get_attr("shape_dtype"))
  _execute.record_gradient(
      "StatefulUniformInt", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

@_doc_controls.do_not_generate_docs
@_kwarg_only
def StatefulUniformInt(resource, algorithm, shape, minval, maxval):
  return stateful_uniform_int(resource=resource, algorithm=algorithm, shape=shape, minval=minval, maxval=maxval)
tf_export("raw_ops.StatefulUniformInt")(StatefulUniformInt)


def stateful_uniform_int_eager_fallback(resource, algorithm, shape, minval, maxval, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function stateful_uniform_int
  """
  _ctx = ctx if ctx else _context.context()
  _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([minval, maxval], _ctx, _dtypes.int64)
  (minval, maxval) = _inputs_dtype
  _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int64)
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
  _inputs_flat = [resource, algorithm, shape, minval, maxval]
  _attrs = ("dtype", _attr_dtype, "shape_dtype", _attr_shape_dtype)
  _result = _execute.execute(b"StatefulUniformInt", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "StatefulUniformInt", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "NonDeterministicInts"
#   input_arg {
#     name: "shape"
#     type_attr: "shape_dtype"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#   }
#   attr {
#     name: "shape_dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "StatefulStandardNormal"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "shape"
#     type_attr: "shape_dtype"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#   }
#   attr {
#     name: "shape_dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "StatefulStandardNormalV2"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "algorithm"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "shape"
#     type_attr: "shape_dtype"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#   }
#   attr {
#     name: "shape_dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "StatefulUniformFullInt"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "algorithm"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "shape"
#     type_attr: "shape_dtype"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_UINT64
#     }
#   }
#   attr {
#     name: "shape_dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "StatefulUniformInt"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "algorithm"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "shape"
#     type_attr: "shape_dtype"
#   }
#   input_arg {
#     name: "minval"
#     type_attr: "dtype"
#   }
#   input_arg {
#     name: "maxval"
#     type_attr: "dtype"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#   }
#   attr {
#     name: "shape_dtype"
#     type: "type"
#     default_value {
#       type: DT_INT64
#     }
#   }
#   is_stateful: true
# }
_op_def_lib = _InitOpDefLibrary(b"\nl\n\024NonDeterministicInts\022\024\n\005shape\"\013shape_dtype\032\017\n\006output\"\005dtype\"\021\n\005dtype\022\004type\032\0020\t\"\027\n\013shape_dtype\022\004type\032\0020\t\210\001\001\n|\n\026StatefulStandardNormal\022\014\n\010resource\030\024\022\024\n\005shape\"\013shape_dtype\032\017\n\006output\"\005dtype\"\021\n\005dtype\022\004type\032\0020\001\"\027\n\013shape_dtype\022\004type\032\0020\t\210\001\001\n\215\001\n\030StatefulStandardNormalV2\022\014\n\010resource\030\024\022\r\n\talgorithm\030\t\022\024\n\005shape\"\013shape_dtype\032\017\n\006output\"\005dtype\"\021\n\005dtype\022\004type\032\0020\001\"\027\n\013shape_dtype\022\004type\032\0020\t\210\001\001\n\213\001\n\026StatefulUniformFullInt\022\014\n\010resource\030\024\022\r\n\talgorithm\030\t\022\024\n\005shape\"\013shape_dtype\032\017\n\006output\"\005dtype\"\021\n\005dtype\022\004type\032\0020\027\"\027\n\013shape_dtype\022\004type\032\0020\t\210\001\001\n\251\001\n\022StatefulUniformInt\022\014\n\010resource\030\024\022\r\n\talgorithm\030\t\022\024\n\005shape\"\013shape_dtype\022\017\n\006minval\"\005dtype\022\017\n\006maxval\"\005dtype\032\017\n\006output\"\005dtype\"\021\n\005dtype\022\004type\032\0020\t\"\027\n\013shape_dtype\022\004type\032\0020\t\210\001\001")
