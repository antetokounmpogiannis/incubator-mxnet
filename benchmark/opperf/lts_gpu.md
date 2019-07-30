# Runtime Features
0. BLAS_APPLE : ✖ BLAS_APPLE
0. BLAS_ATLAS : ✖ BLAS_ATLAS
0. BLAS_MKL : ✖ BLAS_MKL
0. BLAS_OPEN : ✔ BLAS_OPEN
0. CAFFE : ✖ CAFFE
0. CPU_AVX : ✔ CPU_AVX
0. CPU_AVX2 : ✖ CPU_AVX2
0. CPU_SSE : ✔ CPU_SSE
0. CPU_SSE2 : ✔ CPU_SSE2
0. CPU_SSE3 : ✔ CPU_SSE3
0. CPU_SSE4A : ✖ CPU_SSE4A
0. CPU_SSE4_1 : ✔ CPU_SSE4_1
0. CPU_SSE4_2 : ✔ CPU_SSE4_2
0. CUDA : ✔ CUDA
0. CUDA_RTC : ✔ CUDA_RTC
0. CUDNN : ✔ CUDNN
0. CXX14 : ✖ CXX14
0. DEBUG : ✔ DEBUG
0. DIST_KVSTORE : ✖ DIST_KVSTORE
0. F16C : ✔ F16C
0. INT64_TENSOR_SIZE : ✖ INT64_TENSOR_SIZE
0. JEMALLOC : ✔ JEMALLOC
0. LAPACK : ✔ LAPACK
0. MKLDNN : ✔ MKLDNN
0. NCCL : ✖ NCCL
0. OPENCV : ✔ OPENCV
0. OPENMP : ✔ OPENMP
0. PROFILER : ✖ PROFILER
0. SIGNAL_HANDLER : ✔ SIGNAL_HANDLER
0. SSE : ✖ SSE
0. TENSORRT : ✖ TENSORRT
0. TVM_OP : ✖ TVM_OP
# Benchmark Results
| Operator | Avg Forward Time (ms) | Avg. Backward Time (ms) | Max Mem Usage (Storage) (Bytes) | Inputs |
| :---: | :---: | :---: | :---:| :--- |
| FullyConnected | 0.5177 | 0.647 | 4.096 | {'data': (32, 3, 256, 256), 'weight': (64, 196608), 'bias': (64,), 'num_hidden': 64} |
| argmin | 4.59 | --- | 2.048 | {'data': (1024, 1024), 'axis': 0} |
| argmin | 41.073 | --- | 0.002 | {'data': (10000, 1), 'axis': 0} |
| argmin | 42.5071 | --- | 0.4 | {'data': (10000, 100), 'axis': 0} |
| argsort | 66.5087 | --- | 2097.1521 | {'data': (1024, 1024), 'axis': 0} |
| argsort | 14.5439 | --- | 20.0 | {'data': (10000, 1), 'axis': 0} |
| argsort | 56.0861 | --- | 2000.0 | {'data': (10000, 100), 'axis': 0} |
| broadcast_like | 0.1424 | --- | 0.012 | {'lhs': [(1024, 1024), (10000, 10), (10000, 1)], 'rhs': [(1024, 1024), (10000, 10), (10000, 1)]} |
| broadcast_to | 0.6124 | --- | 2097.1521 | {'data': (1024, 1024), 'shape': (1024, 1024)} |
| broadcast_to | 0.1738 | --- | 20.0 | {'data': (10000, 1), 'shape': (10000, 1)} |
| broadcast_to | 0.5926 | --- | 2000.0 | {'data': (10000, 100), 'shape': (10000, 100)} |
| clip | 0.1679 | 0.1552 | 2097.1521 | {'data': (1024, 1024), 'a_min': 1, 'a_max': 8} |
| clip | 0.148 | 0.1316 | 20.0 | {'data': (10000, 1), 'a_min': 1, 'a_max': 8} |
| clip | 0.1629 | 0.147 | 2000.0 | {'data': (10000, 100), 'a_min': 1, 'a_max': 8} |
| depth_to_space | 0.1607 | --- | 0.064 | {'data': (1, 4, 2, 4), 'block_size': 2} |
| depth_to_space | 0.3165 | --- | 500.0 | {'data': (10, 25, 10, 100), 'block_size': 5} |
| expand_dims | 0.1508 | --- | 2097.1521 | {'data': (1024, 1024), 'axis': 0} |
| expand_dims | 0.1483 | --- | 40.0 | {'data': (10000, 1), 'axis': 0} |
| expand_dims | 0.1766 | --- | 4000.0 | {'data': (10000, 100), 'axis': 0} |
| flip | 0.1939 | --- | 0.064 | {'data': (1, 4, 2, 4), 'axis': 0} |
| flip | 0.2146 | --- | 1000.0 | {'data': (10, 25, 10, 100), 'axis': 0} |
| ones_like | 0.1479 | --- | 2097.1521 | {'data': (1024, 1024)} |
| ones_like | 0.1392 | --- | 20.0 | {'data': (10000, 1)} |
| ones_like | 0.1583 | --- | 2000.0 | {'data': (10000, 100)} |
| pick | 0.1802 | 0.3235 | 2.048 | {'data': (1024, 1024), 'index': (1024,), 'axis': 0} |
| pick | 0.156 | 0.1589 | 0.002 | {'data': (10000, 1), 'index': (1,), 'axis': 0} |
| pick | 0.1796 | 0.32 | 0.2 | {'data': (10000, 100), 'index': (100,), 'axis': 0} |
| random_randint | 4.2877 | --- | 2097.1521 | {'low': 0, 'high': 5, 'shape': (1024, 1024)} |
| random_randint | 0.123 | --- | 20.0 | {'low': 0, 'high': 5, 'shape': (10000, 1)} |
| random_randint | 3.4407 | --- | 2000.0 | {'low': 0, 'high': 5, 'shape': (10000, 100)} |
| random_uniform | 10.7681 | --- | 2097.1521 | {'low': 0, 'high': 5, 'shape': (1024, 1024)} |
| random_uniform | 0.2019 | --- | 20.0 | {'low': 0, 'high': 5, 'shape': (10000, 1)} |
| random_uniform | 9.4946 | --- | 2000.0 | {'low': 0, 'high': 5, 'shape': (10000, 100)} |
| softmax | 0.3572 | 0.2877 | 2097.1521 | {'data': (1024, 1024), 'axis': 0} |
| softmax | 0.5944 | 0.3646 | 20.0 | {'data': (10000, 1), 'axis': 0} |
| softmax | 0.8416 | 0.4432 | 2000.0 | {'data': (10000, 100), 'axis': 0} |
| sort | 67.1739 | --- | 4194.3042 | {'data': (1024, 1024), 'axis': 0} |
| sort | 14.4445 | --- | 60.0 | {'data': (10000, 1), 'axis': 0} |
| sort | 57.2142 | --- | 6000.0 | {'data': (10000, 100), 'axis': 0} |
| space_to_depth | 0.1143 | --- | 0.064 | {'data': (1, 4, 2, 4), 'block_size': 2} |
| space_to_depth | 0.1642 | --- | 500.0 | {'data': (10, 25, 10, 100), 'block_size': 5} |
| split | --- | --- | 3145.728 | {'data': (1024, 1024), 'num_outputs': 2, 'axis': 0} |
| split | --- | --- | 30.0 | {'data': (10000, 1), 'num_outputs': 2, 'axis': 0} |
| split | --- | --- | 3000.0 | {'data': (10000, 100), 'num_outputs': 2, 'axis': 0} |
| swapaxes | 0.1142 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 0, 'dim2': 1} |
| swapaxes | 0.1106 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 1, 'dim2': 2} |
| swapaxes | 0.1118 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 2, 'dim2': 3} |
| swapaxes | 0.1111 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 3, 'dim2': 0} |
| take | 0.2209 | 4.026 | 2097.1521 | {'a': (1024, 1024), 'indices': (1024,), 'axis': 0} |
| take | 0.1129 | 0.1244 | 0.002 | {'a': (10000, 1), 'indices': (1,), 'axis': 0} |
| take | 0.1588 | 0.7933 | 20.0 | {'a': (10000, 100), 'indices': (100,), 'axis': 0} |
| tile | 1.4662 | 1.7108 | 4194.3042 | {'data': (1024, 1024), 'reps': (2,)} |
| tile | 0.2386 | 0.1875 | 40.0 | {'data': (10000, 1), 'reps': (2,)} |
| tile | 1.5301 | 1.6533 | 8000.0 | {'data': (10000, 100), 'reps': (2,)} |
| topk | 1.1527 | --- | 2.048 | {'data': (1024, 1024), 'axis': 0, 'k': 1} |
| topk | 0.4053 | --- | 0.002 | {'data': (10000, 1), 'axis': 0, 'k': 1} |
| topk | 1.1134 | --- | 0.2 | {'data': (10000, 100), 'axis': 0, 'k': 1} |
| transpose | 0.1686 | --- | 0.128 | {'data': (1, 4, 2, 4)} |
| transpose | 0.3176 | --- | 1000.0 | {'data': (10, 25, 10, 100)} |
| zeros_like | 0.1343 | --- | 4194.3042 | {'data': (1024, 1024)} |
| zeros_like | 0.1191 | --- | 40.0 | {'data': (10000, 1)} |
| zeros_like | 0.148 | --- | 4000.0 | {'data': (10000, 100)} |