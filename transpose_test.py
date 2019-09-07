from benchmark.opperf.utils.benchmark_utils import run_performance_test
import mxnet as mx
from mxnet import nd
print(run_performance_test(nd.transpose, run_backward=True, dtype='float32', ctx=mx.cpu(),inputs=[{"data":(1024,1024)}],warmup=10, runs=25, profiler='python'))
print(run_performance_test(nd.transpose, run_backward=True, dtype='float32', ctx=mx.cpu(),inputs=[{"data":(10000,10000)}],warmup=10, runs=25, profiler='python'))
