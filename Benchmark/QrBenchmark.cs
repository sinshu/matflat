﻿using System;
using System.Linq;
using BenchmarkDotNet.Attributes;

namespace Benchmark
{
    [MemoryDiagnoser]
    //[ShortRunJob]
    public unsafe class QrBenchmark
    {
        private double[] values;
        private double[] a;
        private double[] q;
        private double[] r;
        private global::MathNet.Numerics.Providers.LinearAlgebra.ManagedLinearAlgebraProvider mathNetProvider;
        private double[] mathNetTau;
        private double[] matFlatRdiag;
        private double[] openBlasTau;

        [Params(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)]
        public int Order;

        [GlobalSetup]
        public void Setup()
        {
            var random = new Random(42);
            values = Enumerable.Range(0, Order * Order).Select(i => random.NextDouble()).ToArray();
            a = new double[values.Length];
            q = new double[values.Length];
            r = new double[values.Length];
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
            fixed (double* pq = q)
            fixed (double* pr = r)
            {
                global::MatFlat.Factorization.Qr(Order, Order, pa, Order, prdiag);
                global::MatFlat.Factorization.QrOrthogonalFactor(Order, Order, pa, Order, pq, Order);
                global::MatFlat.Factorization.QrUpperTriangularFactor(Order, Order, pa, Order, pr, Order, prdiag);
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
                global::OpenBlasSharp.Lapack.Dorgqr(global::OpenBlasSharp.MatrixLayout.ColMajor, Order, Order, Order, pa, Order, ptau);
            }
        }
    }
}
