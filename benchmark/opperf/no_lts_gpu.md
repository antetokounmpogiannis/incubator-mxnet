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
| FullyConnected | 0.4972 | 0.6285 | 4.096 | {'data': (32, 3, 256, 256), 'weight': (64, 196608), 'bias': (64,), 'num_hidden': 64} |
| argmin | 4.923 | --- | 2.048 | {'data': (1024, 1024), 'axis': 0} |
| argmin | 41.1774 | --- | 0.002 | {'data': (10000, 1), 'axis': 0} |
| argmin | 42.3845 | --- | 0.2 | {'data': (10000, 100), 'axis': 0} |
| argsort | 66.589 | --- | 2097.1521 | {'data': (1024, 1024), 'axis': 0} |
| argsort | 14.4682 | --- | 20.0 | {'data': (10000, 1), 'axis': 0} |
| argsort | 56.082 | --- | 2000.0 | {'data': (10000, 100), 'axis': 0} |
| broadcast_like | 0.3026 | --- | 0.036 | {'lhs': [(1024, 1024), (10000, 10), (10000, 1)], 'rhs': [(1024, 1024), (10000, 10), (10000, 1)]} |
| broadcast_to | 0.6396 | --- | 4194.3042 | {'data': (1024, 1024), 'shape': (1024, 1024)} |
| broadcast_to | 0.1959 | --- | 40.0 | {'data': (10000, 1), 'shape': (10000, 1)} |
| broadcast_to | 0.6118 | --- | 4000.0 | {'data': (10000, 100), 'shape': (10000, 100)} |
| clip | 0.1701 | 0.1477 | 2097.1521 | {'data': (1024, 1024), 'a_min': 1, 'a_max': 8} |
| clip | 0.154 | 0.1356 | 20.0 | {'data': (10000, 1), 'a_min': 1, 'a_max': 8} |
| clip | 0.1653 | 0.1489 | 2000.0 | {'data': (10000, 100), 'a_min': 1, 'a_max': 8} |
| depth_to_space | 0.1469 | --- | 0.064 | {'data': (1, 4, 2, 4), 'block_size': 2} |
| depth_to_space | 0.3073 | --- | 500.0 | {'data': (10, 25, 10, 100), 'block_size': 5} |
| expand_dims | 0.1519 | --- | 4194.3042 | {'data': (1024, 1024), 'axis': 0} |
| expand_dims | 0.1364 | --- | 20.0 | {'data': (10000, 1), 'axis': 0} |
| expand_dims | 0.1726 | --- | 4000.0 | {'data': (10000, 100), 'axis': 0} |
| flip | 0.1996 | --- | 0.128 | {'data': (1, 4, 2, 4), 'axis': 0} |
| flip | 0.212 | --- | 1000.0 | {'data': (10, 25, 10, 100), 'axis': 0} |
| ones_like | 0.1666 | --- | 4194.3042 | {'data': (1024, 1024)} |
| ones_like | 0.1575 | --- | 40.0 | {'data': (10000, 1)} |
| ones_like | 0.1609 | --- | 4000.0 | {'data': (10000, 100)} |
| pick | 0.1868 | 0.3235 | 2.048 | {'data': (1024, 1024), 'index': (1024,), 'axis': 0} |
| pick | 0.1652 | 0.1617 | 0.002 | {'data': (10000, 1), 'index': (1,), 'axis': 0} |
| pick | 0.16 | 0.3155 | 0.2 | {'data': (10000, 100), 'index': (100,), 'axis': 0} |
| random_randint | 4.2212 | --- | 4194.3042 | {'low': 0, 'high': 5, 'shape': (1024, 1024)} |
| random_randint | 0.1231 | --- | 20.0 | {'low': 0, 'high': 5, 'shape': (10000, 1)} |
| random_randint | 3.4828 | --- | 4000.0 | {'low': 0, 'high': 5, 'shape': (10000, 100)} |
| random_uniform | 10.512 | --- | 4194.3042 | {'low': 0, 'high': 5, 'shape': (1024, 1024)} |
| random_uniform | 0.2166 | --- | 20.0 | {'low': 0, 'high': 5, 'shape': (10000, 1)} |
| random_uniform | 9.7422 | --- | 4000.0 | {'low': 0, 'high': 5, 'shape': (10000, 100)} |
| softmax | 0.3691 | 0.2964 | 2097.1521 | {'data': (1024, 1024), 'axis': 0} |
| softmax | 0.6075 | 0.3882 | 20.0 | {'data': (10000, 1), 'axis': 0} |
| softmax | 0.8469 | 0.4618 | 2000.0 | {'data': (10000, 100), 'axis': 0} |
| sort | 67.1776 | --- | 4194.3042 | {'data': (1024, 1024), 'axis': 0} |
| sort | 14.3759 | --- | 40.0 | {'data': (10000, 1), 'axis': 0} |
| sort | 56.7906 | --- | 6000.0 | {'data': (10000, 100), 'axis': 0} |
| space_to_depth | 0.0933 | --- | 0.064 | {'data': (1, 4, 2, 4), 'block_size': 2} |
| space_to_depth | 0.1491 | --- | 500.0 | {'data': (10, 25, 10, 100), 'block_size': 5} |
| split | --- | --- | 3145.728 | {'data': (1024, 1024), 'num_outputs': 2, 'axis': 0} |
| split | --- | --- | 30.0 | {'data': (10000, 1), 'num_outputs': 2, 'axis': 0} |
| split | --- | --- | 3000.0 | {'data': (10000, 100), 'num_outputs': 2, 'axis': 0} |
| swapaxes | 0.1162 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 0, 'dim2': 1} |
| swapaxes | 0.1145 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 1, 'dim2': 2} |
| swapaxes | 0.124 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 2, 'dim2': 3} |
| swapaxes | 0.1138 | --- | 0.064 | {'data': (1, 4, 2, 4), 'dim1': 3, 'dim2': 0} |
| take | 0.1986 | 4.0508 | 2097.1521 | {'a': (1024, 1024), 'indices': (1024,), 'axis': 0} |
| take | 0.097 | 0.1005 | 0.002 | {'a': (10000, 1), 'indices': (1,), 'axis': 0} |
| take | 0.1031 | 0.7786 | 20.0 | {'a': (10000, 100), 'indices': (100,), 'axis': 0} |
| tile | 1.4587 | 1.7211 | 4194.3042 | {'data': (1024, 1024), 'reps': (2,)} |
| tile | 0.1866 | 0.1404 | 40.0 | {'data': (10000, 1), 'reps': (2,)} |
| tile | 1.39 | 1.6232 | 4000.0 | {'data': (10000, 100), 'reps': (2,)} |
| topk | 1.0958 | --- | 2.048 | {'data': (1024, 1024), 'axis': 0, 'k': 1} |
| topk | 0.3392 | --- | 0.002 | {'data': (10000, 1), 'axis': 0, 'k': 1} |
| topk | 1.0911 | --- | 0.2 | {'data': (10000, 100), 'axis': 0, 'k': 1} |
| transpose | 0.171 | --- | 0.064 | {'data': (1, 4, 2, 4)} |
| transpose | 0.3031 | --- | 500.0 | {'data': (10, 25, 10, 100)} |
| zeros_like | 0.0994 | --- | 2097.1521 | {'data': (1024, 1024)} |
| zeros_like | 0.0729 | --- | 20.0 | {'data': (10000, 1)} |
| zeros_like | 0.0768 | --- | 2000.0 | {'data': (10000, 100)} |