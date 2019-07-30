#!/usr/bin/env python3
#
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
#
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

import mxnet as mx
from mxnet import gluon, nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test, run_op_benchmarks
from benchmark.opperf.opperf import _parse_mxnet_context
from benchmark.opperf.utils.common_utils import save_to_file
from benchmark.opperf.utils.op_registry_utils import get_current_runtime_features, get_all_large_tensor_operators

# dimension constants
# MEDIUM_X = 10000
# LARGE_X = 100000000
# LARGE_Y = 50000000
# SMALL_X = 1024
# SMALL_Y = 50
# LARGE_SIZE = LARGE_X * SMALL_Y
# inputs = {"shape":(SMALL_X, SMALL_Y),"low":0}
def run_large_test_benchmarks(ctx = mx.cpu(), dtype='float32'):
    # Fetch all Large tensor Operators
    mx_large_tensor_ops = get_all_large_tensor_operators()
    # Run benchmarks
    mx_large_tensor_results = run_op_benchmarks(mx_large_tensor_ops, dtype, ctx, warmup=10, runs=25)
    # mx_large_tensor_results = run_performance_test([nd.zeros, nd.ones , nd.empty, nd.random.uniform, nd.random.randint, nd.dot, nd.split,
    #                     nd.clip, nd.FullyConnected, nd.broadcast_to, nd.broadcast_like, nd.tile, nd.take,
    #                     nd.slice, nd.expand_dims, nd.sort, nd.argsort, nd.argmin, nd.topk, nd.squeeze, nd.where,
    #                     nd.sparse.where, nd.pick, nd.depth_to_space, nd.space_to_depth, nd.diag, nd.ravel_multi_index,
    #                     nd.unravel_index, nd.swapaxes, nd.flip, nd.softmax],
    #                     run_backward=False,
    #                     dtype=dtype,
    #                     ctx=ctx,
    #                     inputs=[inputs],
    #                     warmup=10,
    #                     runs=25)
    # not able to use run_perfr_test since all ops should have same inputs, else it gives zeros doesnt hv param low, high etc
    return mx_large_tensor_results

def main():
    # 0. LARGE TENSOR FLAG
    run_time_features = get_current_runtime_features()
    if(str(run_time_features['runtime_features']['INT64_TENSOR_SIZE'])=="✔ INT64_TENSOR_SIZE"):
        print("Large tensor support : ON")
    else:
        print("Large tensor support : OFF")

    # 1. GET USER INPUTS
    parser = argparse.ArgumentParser(description='Run large tensor benchmarks')

    parser.add_argument('--ctx', type=str, default='cpu',
                        help='Global context to run all benchmarks. By default, cpu on a '
                             'CPU machine, gpu(0) on a GPU machine. '
                             'Valid Inputs - cpu, gpu, gpu(0), gpu(1)...')
    parser.add_argument('--dtype', type=str, default='float32', help='DType (Precision) to run benchmarks. By default, '
                                                                     'float32. Valid Inputs - float32, float64, int32, '
                                                                     'int64')
    # parser.add_argument('-f', '--output-format', type=str, default='json',
    #                     choices=['json', 'md'],
    #                     help='Benchmark result output format. By default, json. '
    #                          'Valid Inputs - json, md')

    # parser.add_argument('-o', '--output-file', type=str, default='./mxnet_operator_benchmarks.json',
    #                     help='Name and path for the '
    #                          'output file.')

    args = parser.parse_args()
    # logging.info("Running Large tensor benchmarks with the following options: {args}".format(args=args))
    # assert not os.path.isfile(args.output_file),\
    #     "Output file {output_file} already exists.".format(output_file=args.output_file)

    # 2. RUN BENCHMARKS
    ctx = _parse_mxnet_context(args.ctx)
    dtype = args.dtype
    final_benchmark_results = run_large_test_benchmarks(ctx=ctx, dtype=dtype)
    print(final_benchmark_results)
    # 3. SAVE OUTPUTS
    # save_to_file(final_benchmark_results, args.output_file, args.output_format, run_time_features)

    return 0


if __name__ == '__main__':
    sys.exit(main())
