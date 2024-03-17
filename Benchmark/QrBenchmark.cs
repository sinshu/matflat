using System;
using System.Linq;
using BenchmarkDotNet.Attributes;

namespace Benchmark
{
    [MemoryDiagnoser]
    [ShortRunJob]
    public unsafe class QrBenchmark
    {
        private double[] values;
        private double[] a;
        private double[] q;
        private global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider mathNetProvider;
        private double[] mathNetTau;
        private double[] matFlatRdiag;
        private double[] openBlasTau;

        [Params(5, 10, 20, 50, 100, 200)]
        public int Order;

        [GlobalSetup]
        public void Setup()
        {
            var random = new Random(42);
            values = Enumerable.Range(0, Order * Order).Select(i => random.NextDouble()).ToArray();
            a = new double[values.Length];
            q = new double[values.Length];
            mathNetProvider = global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider.Instance;
            mathNetTau = new double[Order];
            matFlatRdiag = new double[Order];
            openBlasTau = new double[Order];
        }

        [Benchmark]
        public void MathNet()
        {
            values.CopyTo(a, 0);

            mathNetProvider.QRFactor(a, Order, Order, q, mathNetTau);
        }

        [Benchmark]
        public void MatFlat()
        {
            values.CopyTo(a, 0);

            fixed (double* pa = a)
            fixed (double* prdiag = matFlatRdiag)
            {
                global::MatFlat.Factorization.QrDouble(Order, Order, pa, Order, prdiag);
            }
        }

        [Benchmark]
        public void OpenBlas()
        {
            values.CopyTo(a, 0);

            fixed (double* pa = a)
            fixed (double* ptau = openBlasTau)
            {
                global::OpenBlasSharp.Lapack.Dgeqrf(global::OpenBlasSharp.MatrixLayout.ColMajor, Order, Order, pa, Order, ptau);
            }
        }
    }
}
