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

import mxnet as mx
from mxnet import gluon, nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test

# dimension constants
MEDIUM_X = 10000
LARGE_X = 100000000
LARGE_Y = 50000000
SMALL_X = 1024
SMALL_Y = 50
LARGE_SIZE = LARGE_X * SMALL_Y

for i in ['cpu','gpu']:
    for j in ['small','large']:
        if i == 'cpu':
            ctx = mx.cpu()
        else:
            ctx = mx.gpu()
        if j == 'small':
            x,y = SMALL_X,SMALL_Y
        else:
            x,y = LARGE_X,LARGE_Y
        run_performance_test([nd.zeros],
                            run_backward=True,
                            dtype='float32',
                            ctx=ctx,
                            inputs=[{"shape":(x, y)}],
                            warmup=10,
                            runs=25)