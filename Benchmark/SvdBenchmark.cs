using System;
using System.Linq;
using BenchmarkDotNet.Attributes;

namespace Benchmark
{
    [MemoryDiagnoser]
    //[ShortRunJob]
    public unsafe class SvdBenchmark
    {
        private double[] values;
        private double[] a;
        private double[] s;
        private double[] u;
        private double[] vt;
        private double[] work;
        private global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider mathNetProvider;

        [Params(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)]
        public int Order;

        [GlobalSetup]
        public void Setup()
        {
            var random = new Random(42);
            values = Enumerable.Range(0, Order * Order).Select(i => random.NextDouble()).ToArray();
            a = new double[values.Length];
            s = new double[Order];
            u = new double[values.Length];
            vt = new double[values.Length];
            work = new double[Order];
            mathNetProvider = global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider.Instance;
        }

        [Benchmark]
        public void MathNet()
        {
            values.CopyTo(a, 0);

            mathNetProvider.SingularValueDecomposition(true, a, Order, Order, s, u, vt);
        }

        [Benchmark]
        public void MatFlat()
        {
            values.CopyTo(a, 0);

            fixed (double* pa = a)
            fixed (double* ps = s)
            fixed (double* pu = u)
            fixed (double* pvt = vt)
            {
                global::MatFlat.Factorization.Svd(Order, Order, pa, Order, ps, pu, Order, pvt, Order);
            }
        }

        [Benchmark]
        public void OpenBlas()
        {
            values.CopyTo(a, 0);

            fixed (double* pa = a)
            fixed (double* ps = s)
            fixed (double* pu = u)
            fixed (double* pvt = vt)
            fixed (double* pwork = work)
            {
                global::OpenBlasSharp.Lapack.Dgesvd(
                    global::OpenBlasSharp.MatrixLayout.ColMajor,
                    'A', 'A',
                    Order, Order,
                    pa, Order,
                    ps,
                    pu, Order,
                    pvt, Order,
                    pwork);
            }
        }
    }
}
