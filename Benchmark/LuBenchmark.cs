using System;
using System.Linq;
using BenchmarkDotNet.Attributes;

namespace Benchmark
{
    [MemoryDiagnoser]
    //[ShortRunJob]
    public unsafe class LuBenchmark
    {
        private double[] values;
        private double[] a;
        private global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider mathNetProvider;
        private int[] mathNetPiv;
        private int[] matFlatPiv;
        private int[] openBlasPiv;

        [Params(5, 10, 20, 50, 100, 200)]
        public int Order;

        [GlobalSetup]
        public void Setup()
        {
            var random = new Random(42);
            values = Enumerable.Range(0, Order * Order).Select(i => random.NextDouble()).ToArray();
            a = new double[values.Length];
            mathNetProvider = global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider.Instance;
            mathNetPiv = new int[Order];
            matFlatPiv = new int[Order];
            openBlasPiv = new int[Order];
        }

        [Benchmark]
        public void MathNet()
        {
            values.CopyTo(a, 0);

            mathNetProvider.LUFactor(a, Order, mathNetPiv);
        }

        [Benchmark]
        public void MatFlat()
        {
            values.CopyTo(a, 0);

            int sign = 0;
            fixed (double* pa = a)
            fixed (int* ppiv = matFlatPiv)
            {
                global::MatFlat.Factorization.Lu(Order, Order, pa, Order, ppiv, &sign);
            }
        }

        //[Benchmark]
        public void OpenBlas()
        {
            values.CopyTo(a, 0);

            fixed (double* pa = a)
            fixed (int* ppiv = openBlasPiv)
            {
                global::OpenBlasSharp.Lapack.Dgetrf(global::OpenBlasSharp.MatrixLayout.ColMajor, Order, Order, pa, Order, ppiv);
            }
        }
    }
}
