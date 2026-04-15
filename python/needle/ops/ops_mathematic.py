"""Operator implementations."""

import array
from numbers import Number
from re import I
from tkinter import NO
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a, b = node.inputs
        grad_a = out_grad * b * (power(a, b - 1))
        grad_b = out_grad * power(a, b) * log(a)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * (power_scalar(a, self.scalar - 1))


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0]
        return out_grad * self.scalar * (power_scalar(a, self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


# a/b
class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a, b = node.inputs
        return out_grad / b, out_grad * negate(a) / (b * b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


# 3*2 to 2*3
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        return array_api.swapaxes(a, -1, -2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        initial_shape = node.inputs[0].shape
        # 把上游传回来的梯度 reshape 回去
        return reshape(out_grad, initial_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape

        # 如果形状相同，无需操作
        if input_shape == self.shape:
            return out_grad

        # 找出需要求和的轴
        axes_to_sum = []
        # 对齐维度：从后往前比较
        # 例如: input (1, 3), output (2, 3)
        # i=1: input[1]=3, out[1]=3 -> 匹配
        # i=0: input[0]=1, out[0]=2 -> 不匹配，需累加
        for i in range(len(self.shape)):
            # 对应的输入维度索引
            input_idx = i - (len(self.shape) - len(input_shape))

            # 如果输入维度不存在（即被 prepend 了 1），或者输入维度是 1 但输出不是 1
            if input_idx < 0 or (input_shape[input_idx] == 1 and self.shape[i] != 1):
                axes_to_sum.append(i)

        if axes_to_sum:
            # 关键：使用 keepdims=True 保持维度数量不变，这样结果形状自动对齐 input_shape 的尾部
            # 但注意：如果 input 是 (3,) 而 output 是 (2, 3)，sum(axis=0, keepdims=True) 得到 (1, 3)。
            # 这时还需要 reshape 吗？
            # 其实，如果 Summation 返回的形状是 (1, 3)，而我们需要 (3,)，还是得 reshape。

            # 所以，最通用的写法依然是：Sum + Reshape
            grad = summation(out_grad, axes=tuple(axes_to_sum))
            # 由于 summation 默认可能不 keepdims，我们需要 reshape 回原始形状
            return reshape(grad, input_shape)
        else:
            return out_grad


# class BroadcastTo(TensorOp):
#     def __init__(self, shape):
#         self.shape = shape

#     def compute(self, a):
#         ### BEGIN YOUR SOLUTION
#         # raise NotImplementedError()
#         return array_api.broadcast_to(a, self.shape)

#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         # raise NotImplementedError()
#         input_shape = node.inputs[0].shape
#         if input_shape == self.shape:
#             return out_grad

#         ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        initial_shape = node.inputs[0].shape
        new_shape = list(initial_shape)

        if self.axes is not None:
            axes = self.axes

            if isinstance(axes, int):
                axes = (axes,)

            for ax in axes:
                new_shape[ax] = 1

        else:
            # 如果axex是none 说明全是标量 把他全部变成 1
            new_shape = [1] * len(initial_shape)
        return broadcast_to(reshape(out_grad, tuple(new_shape)), initial_shape)

        # input_shape = node.inputs[0].shape
        # return broadcast_to(out_grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        # 1. 计算未考虑广播的梯度
        # dL/dA = dL/dOut @ B.T
        grad_lhs = matmul(out_grad, transpose(rhs))
        # dL/dB = A.T @ dL/dOut
        grad_rhs = matmul(transpose(lhs), out_grad)

        # 2. 处理广播机制 (Broadcasting)
        # 如果 lhs 或 rhs 在 batch 维度上是 1，而 out_grad 的 batch 维度 > 1，
        # 那么梯度需要在相应的 batch 维度上求和。

        def reduce_broadcast(grad, original_shape):
            """
            将 grad 的形状缩减回 original_shape。
            通过在对齐后多出来的维度或原始形状为1的维度上求和来实现。
            """
            current_shape = grad.shape
            if current_shape == original_shape:
                return grad

            # 计算需要求和的轴
            axes_to_sum = []
            # 从后往前对齐维度
            len_curr = len(current_shape)
            len_orig = len(original_shape)

            for i in range(len_curr):
                # 对应的原始维度索引
                orig_idx = i - (len_curr - len_orig)

                # 情况1: 当前维度在原始形状中不存在（即前置广播维度）
                if orig_idx < 0:
                    axes_to_sum.append(i)
                # 情况2: 当前维度在原始形状中存在，但原始形状是1，当前形状>1
                elif original_shape[orig_idx] == 1 and current_shape[i] != 1:
                    axes_to_sum.append(i)

            if axes_to_sum:
                # 对需要缩减的轴求和，并保持维度（keepdims=True 的逻辑通过 reshape 实现）
                # 注意：我们的 summation 默认不 keepdims，所以求和后形状会变
                # 为了安全，我们先求和，再 reshape 回原始形状
                grad_reduced = summation(grad, axes=tuple(axes_to_sum))
                # 由于 summation 去掉了维度，我们需要 reshape 回去
                # 但要注意：如果 original_shape 是 (1, 3)，summation(axis=0) 得到 (3,)
                # reshape((3,), (1, 3)) 是合法的
                return reshape(grad_reduced, original_shape)
            else:
                return grad

        grad_lhs_final = reduce_broadcast(grad_lhs, lhs.shape)
        grad_rhs_final = reduce_broadcast(grad_rhs, rhs.shape)

        return grad_lhs_final, grad_rhs_final
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()

        a = node.inputsp[0]
        return out_grad * Tensor(a.numpy() > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
