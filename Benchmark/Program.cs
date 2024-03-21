using System;
using BenchmarkDotNet.Running;

namespace Benchmark
{
    internal class Program
    {
        static void Main(string[] args)
        {
            BenchmarkRunner.Run<LuBenchmark>();
            BenchmarkRunner.Run<CholeskyBenchmark>();
            BenchmarkRunner.Run<QrBenchmark>();
            BenchmarkRunner.Run<SvdBenchmark>();
        }
    }
}
