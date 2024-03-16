using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using OpenBlasSharp;

namespace Benchmark
{
    [MemoryDiagnoser]
    [ShortRunJob]
    public unsafe class CholeskyBenchmark
    {
        private double[] values;
        private double[] a;
        private global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider mathNetProvider;

        [Params(5, 10, 20, 50)]
        public int Order;

        [GlobalSetup]
        public void Setup()
        {
            values = GetDecomposableDouble(42, Order);
            a = new double[values.Length];
            mathNetProvider = global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider.Instance;
        }

        [Benchmark]
        public void MathNet()
        {
            values.CopyTo(a, 0);

            mathNetProvider.CholeskyFactor(a, Order);
        }

        [Benchmark]
        public void MatFlat()
        {
            values.CopyTo(a, 0);

            fixed (double* pa = a)
            {
                global::MatFlat.Factorization.CholeskyDouble(Order, pa, Order);
            }
        }

        [Benchmark]
        public void OpenBlas()
        {
            values.CopyTo(a, 0);

            fixed (double* pa = a)
            {
                global::OpenBlasSharp.Lapack.Dpotrf(global::OpenBlasSharp.MatrixLayout.ColMajor, 'U', Order, pa, Order);
            }
        }

        private static unsafe double[] GetDecomposableDouble(int seed, int n)
        {
            var random = new Random(42);
            var a = Enumerable.Range(0, n * n).Select(i => random.NextDouble()).ToArray();

            var symmetric = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = symmetric)
            {
                Blas.Dgemm(
                    global::OpenBlasSharp.Order.ColMajor,
                    Transpose.NoTrans,
                    Transpose.Trans,
                    n, n, n,
                    1.0,
                    pa, n,
                    pa, n,
                    0.0,
                    ps, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = symmetric[n * col + row];
                    a[n * col + row] = value;
                }
            }

            return a;
        }
    }
}
