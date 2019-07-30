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
from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.opperf import _parse_mxnet_context
from benchmark.opperf.utils.common_utils import save_to_file
from benchmark.opperf.utils.op_registry_utils import get_current_runtime_features

# dimension constants
MEDIUM_X = 10000
LARGE_X = 100000000
LARGE_Y = 50000000
SMALL_X = 1024
SMALL_Y = 50
LARGE_SIZE = LARGE_X * SMALL_Y

def run_large_test_benchmarks():
    for i in ['cpu','gpu']:
        if i == 'cpu':
            ctx = mx.cpu()
        else:
            ctx = mx.gpu()
        result = run_performance_test([nd.zeros, nd.ones],
                            run_backward=False,
                            dtype='float32',
                            ctx=ctx,
                            inputs=[{"shape":(SMALL_X, SMALL_Y)}],
                            warmup=10,
                            runs=25)
        return result

def main():
    # 1. GET USER INPUTS
    parser = argparse.ArgumentParser(description='Run large tensor benchmarks')

    # parser.add_argument('--ctx', type=str, default='cpu',
    #                     help='Global context to run all benchmarks. By default, cpu on a '
    #                          'CPU machine, gpu(0) on a GPU machine. '
    #                          'Valid Inputs - cpu, gpu, gpu(0), gpu(1)...')
    # parser.add_argument('--dtype', type=str, default='float32', help='DType (Precision) to run benchmarks. By default, '
    #                                                                  'float32. Valid Inputs - float32, float64, int32, '
    #                                                                  'int64')
    parser.add_argument('-f', '--output-format', type=str, default='json',
                        choices=['json', 'md'],
                        help='Benchmark result output format. By default, json. '
                             'Valid Inputs - json, md')

    parser.add_argument('-o', '--output-file', type=str, default='./mxnet_operator_benchmarks.json',
                        help='Name and path for the '
                             'output file.')

    args = parser.parse_args()
    logging.info("Running Large tensor benchmarks with the following options: {args}".format(args=args))
    assert not os.path.isfile(args.output_file),\
        "Output file {output_file} already exists.".format(output_file=args.output_file)

    # 2. RUN BENCHMARKS
    # ctx = _parse_mxnet_context(args.ctx)
    # dtype = args.dtype
    final_benchmark_results = run_large_test_benchmarks()

    # 3. PREPARE OUTPUTS
    run_time_features = get_current_runtime_features()
    save_to_file(final_benchmark_results, args.output_file, args.output_format, run_time_features)

    # # 4. Generate list of MXNet operators not covered in benchmarks
    # ops_not_covered = get_operators_with_no_benchmark(final_benchmark_results.keys())
    # for idx, op in enumerate(ops_not_covered):
    #     print("{idx}. {op}".format(idx=idx, op=op))

    return 0


if __name__ == '__main__':
    sys.exit(main())
