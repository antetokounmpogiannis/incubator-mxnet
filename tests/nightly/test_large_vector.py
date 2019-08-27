# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import mxnet as mx

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, create_vector, default_context
from mxnet import gluon, nd
from tests.python.unittest.common import with_seed

# dimension constants
LARGE_X = 5000000000
MEDIUM_X = 1000000000


def test_slice():
    a = nd.ones(LARGE_X)
    res = nd.slice(a, begin=(LARGE_X - MEDIUM_X), end=LARGE_X)
    assert a[0] == 1
    assert res.shape[0] == MEDIUM_X


def test_ndarray_zeros():
    a = nd.zeros(shape=LARGE_X)
    assert a[-1] == 0
    assert a.shape == (LARGE_X,)
    assert a.size == LARGE_X


def test_ndarray_ones():
    a = nd.ones(shape=LARGE_X)
    assert a[-1] == 1
    assert nd.sum(a).asnumpy() == LARGE_X


@with_seed()
def test_ndarray_random_uniform():
    a = nd.random.uniform(shape=LARGE_X)
    assert a[-1] != 0


@with_seed()
def test_ndarray_random_randint():
    a = nd.random.randint(100, 10000, shape=LARGE_X)
    assert a.shape == (LARGE_X,)
    # check if randint can generate value greater than 2**32 (large)
    low_large_value = 2**32
    high_large_value = 2**34
    a = nd.random.randint(low_large_value, high_large_value, dtype=np.int64)
    low = mx.nd.array([low_large_value], dtype='int64')
    high = mx.nd.array([high_large_value], dtype='int64')
    assert a > low  and a < high


def test_ndarray_empty():
    a = nd.empty(LARGE_X)
    assert a.shape == (LARGE_X,)


def test_elementwise():
    a = nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X)
    res = a + b
    assert res[-1].asnumpy() == 2
    res = a + 1
    assert res[-1].asnumpy() == 2
    res = nd.sqrt(a + 8)
    assert res[-1].asnumpy() == 3


def test_reduce():
    a = nd.ones(shape=(LARGE_X, 1))
    assert nd.sum(a).asnumpy() == a.shape[0] * a.shape[1]


def test_clip():
    a = create_vector(LARGE_X)
    res = nd.clip(a, a_min=100, a_max=1000)
    assert np.sum(res[-1].asnumpy() == 1000) == 1


def test_argmin():
    a = create_vector(LARGE_X, dtype=np.float32)
    assert a[0] == 0
    idx = mx.nd.argmin(a, axis=0)
    assert idx[0] == 0
    assert idx.shape[0] == 1


def test_take():
    a = nd.ones(shape=LARGE_X)
    idx = nd.arange(LARGE_X - 1000, LARGE_X)
    res = nd.take(a, idx)
    assert np.sum(res.asnumpy() == 1) == res.shape[0]


def test_slice_assign():
    a = nd.ones(shape=LARGE_X)
    a[LARGE_X-1:LARGE_X] = 1000
    assert np.sum(a[-1].asnumpy() == 1000) == 1


def test_expand_dims():
    a = nd.ones(shape=LARGE_X)
    res = nd.expand_dims(a, axis=0)
    assert res[0][0] == 1
    assert res.shape == (1, a.shape[0])


def test_squeeze():
    a = nd.ones(shape=LARGE_X)
    data = nd.expand_dims(a, axis=0)
    res = nd.squeeze(data)
    assert a[0] == res[0]
    assert res.shape == a.shape


def test_broadcast_div():
    a = nd.ones(shape=LARGE_X)
    b = nd.ones(shape=LARGE_X) * 2
    res = a / b
    assert np.sum(res.asnumpy() == 0.5) == a.shape[0]


def test_Dense(ctx=mx.cpu(0)):
    data = mx.nd.ones(shape=LARGE_X)
    linear = gluon.nn.Dense(2)
    linear.initialize(ctx=ctx)
    res = linear(data)
    res.wait_to_read()
    assert res.shape == (LARGE_X, 2)


def test_argsort():
    b = create_vector(size=LARGE_X)
    s = nd.argsort(b, axis=0, is_ascend=False, dtype=np.int64)
    mx.nd.waitall()
    assert (s[0].asnumpy() == (LARGE_X - 1)).all()


def test_sort():
    b = create_vector(size=LARGE_X)
    s = nd.sort(b, axis=0, is_ascend=False)
    assert np.sum(s[-1].asnumpy() == 0).all()
    s = nd.sort(b, is_ascend=True)
    assert np.sum(s[0].asnumpy() == 0).all()


def test_topk():
    b = create_vector(size=LARGE_X)
    ind = nd.topk(b, k=10, axis=0, dtype=np.int64)
    assert np.sum(ind.asnumpy() == (LARGE_X - 1)) == 1
    ind, val = mx.nd.topk(b, k=3, axis=0, dtype=np.int64, ret_typ="both", is_ascend=False)
    assert np.all(ind == val)
    val = nd.topk(b, k=1, axis=0, dtype=np.int64, ret_typ="value")
    assert val.sum() == (LARGE_X - 1)


# TODO: correctness of dropout
# currently only test for dropout to work
# since testing for correctness involves flakiness issue #14288
def test_dropout():
    shape = (LARGE_X, )
    x = mx.sym.var('data')
    y = mx.sym.Dropout(x, p=1, cudnn_off=True)
    exe = y.simple_bind(ctx=default_context(), data=shape)
    exe.arg_arrays[0][:] = 1
    out = exe.forward(is_train=True)
    out[0].wait_to_read()


# def test_activation():
#     a = mx.nd.ones(LARGE_X)
#     test_x = -2
#     a[-1] = test_x

#     # Hyperbolic tangent (tanh)
#     # y = (exp(x)-exp(-x))/(exp(x)+exp(-x))
#     a = mx.nd.Activation(a, act_type="tanh")
#     tanh_x = (np.exp(-2) - np.exp(2)) / (np.exp(-2) + np.exp(2))
#     assert a[-1] == tanh_x

#     # Recitified Linear Unit (relu)
#     # y = max(x,0)
#     a = mx.nd.Activation(a, act_type="relu")
#     assert a[-1] == 0

#     # Sigmoid
#     # y = x/(1+abs(x))
#     a = mx.nd.Activation(a, act_type="sigmoid")
#     sigmoid_x = 1 / (1 + math.exp(-test_x))
#     assert a[-1] == sigmoid_x

#     # Soft Sign
#     # y = 1/(1+exp(-x))
#     a = mx.nd.Activation(a, act_type="softsign")
#     softsign_x = test_x / (1 + abs(test_x))
#     assert a[-1] == softsign_x


# TODO: correctness of prelu (currently flaky)
# def test_leaky_relu():
#     a = -1*mx.nd.ones(LARGE_X)

#     def test_leaky():
#         res = mx.nd.LeakyReLU(a, act_type="leaky", slope=0.3)
#         assert res[-1][-1].asnumpy() == 0.3*a[-1][-1].asnumpy()

#     def test_elu():
#         res = mx.nd.LeakyReLU(a, act_type="elu", slope=0.3)
#         assert res[-1][-1].asnumpy() == 0.3*(np.exp(a[-1][-1].asnumpy())-1)

#     def test_selu():
#         lam = 1.0507009873554804934193349852946
#         alpha = 1.6732632423543772848170429916717
#         res = mx.nd.LeakyReLU(a, act_type="selu")
#         assert res[-1][-1].asnumpy() == (lam * alpha * (np.exp(a[-1][-1].asnumpy())-1))

#     def test_rrelu():
#         lower = 0.125
#         upper = 0.333999991
#         res = mx.nd.LeakyReLU(a, act_type="rrelu")
#         assert res[-1][-1].asnumpy() == (lower + upper) / 2 * a[-1][-1].asnumpy()

#     test_leaky()
#     test_elu()
#     test_selu()
#     test_rrelu()


# def test_softmax_cross_entropy():
#     # SoftmaxCrossEntropy only accepts 2D data
#     # dtype of input data, mxnet cross entropy set explicitly to float64
#     # numpy implicitly takes care of double precision
#     batch_size = 2
#     num_labels = LARGE_X
#     input_data = mx.nd.ones((batch_size, num_labels), dtype="float64")
#     input_label = mx.nd.zeros((batch_size,), dtype="float64")

#     true_softmax = np.full((batch_size, num_labels), (1 / num_labels))
#     # use 1/batch_size when softmax axis=0
#     # here 1/num_labels since softmax_cross_entropy uses default axis
#     # by default axis=1
#     np_one_hot_label = np.zeros((batch_size, num_labels))
#     np_one_hot_label[:, 0] = 1

#     true_softmax_cross_entropy = np.sum(-np.log(true_softmax) *
#                                         np_one_hot_label)
#     mx_softmax_cross_entropy = mx.nd.softmax_cross_entropy(input_data,
#                                                            input_label,
#                                                            dtype="float64")
#     assert_almost_equal(mx_softmax_cross_entropy.asnumpy(),
#                         true_softmax_cross_entropy, rtol=1e-3, atol=1e-5)


# def test_index_copy():
#     x = mx.nd.zeros(LARGE_X)
#     t = mx.nd.array([-1])
#     index = mx.nd.array([LARGE_X - 1])

#     x = mx.nd.contrib.index_copy(x, index, t)
#     assert x[-1] == t[-1]


if __name__ == '__main__':
    import nose
    nose.runmodule()
