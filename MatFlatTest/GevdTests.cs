using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class GevdTests
    {
        [TestCase(1, 1, 1)]
        [TestCase(1, 3, 2)]
        [TestCase(2, 2, 2)]
        [TestCase(2, 4, 3)]
        [TestCase(3, 3, 3)]
        [TestCase(3, 4, 5)]
        [TestCase(5, 5, 5)]
        [TestCase(5, 7, 8)]
        [TestCase(10, 10, 10)]
        [TestCase(10, 17, 11)]
        public unsafe void GevdDouble(int n, int lda, int ldb)
        {
            var a = GetDecomposableDouble(42, n, lda);
            var b = GetDecomposableDouble(57, n, ldb);

            var v = a.ToArray();
            var l = b.ToArray();
            var w = new double[n];

            NanLowerPart(n, v, lda);
            NanLowerPart(n, l, ldb);

            fixed (double* pv = v)
            fixed (double* pl = l)
            fixed (double* pw = w)
            {
                Factorization.Gevd(n, pv, lda, pl, ldb, pw);
            }

            var left = new double[n];
            var right = new double[n];
            fixed (double* pa = a)
            fixed (double* pb = b)
            fixed (double* pv = v)
            fixed (double* pl = left)
            fixed (double* pr = right)
            {
                for (var j = 0; j < n; j++)
                {
                    OpenBlasSharp.Blas.Dgemv(
                        Order.ColMajor,
                        OpenBlasSharp.Transpose.NoTrans,
                        n, n,
                        1.0,
                        pa, lda,
                        pv + lda * j, 1,
                        0.0,
                        pl, 1);

                    OpenBlasSharp.Blas.Dgemv(
                        Order.ColMajor,
                        OpenBlasSharp.Transpose.NoTrans,
                        n, n,
                        w[j],
                        pb, ldb,
                        pv + lda * j, 1,
                        0.0,
                        pr, 1);

                    Assert.That(left, Is.EqualTo(right).Within(1.0E-11));
                }
            }

            for (var i = 0; i < v.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= lda)
                {
                    Assert.That(v[i], Is.EqualTo(a[i]).Within(0));
                }
            }

            for (var i = 0; i < l.Length; i++)
            {
                var row = i % ldb;
                var col = i / ldb;
                if (row >= ldb)
                {
                    Assert.That(l[i], Is.EqualTo(b[i]).Within(0));
                }
            }
        }

        private static unsafe double[] GetDecomposableDouble(int seed, int n, int lda)
        {
            var a = Matrix.RandomDouble(seed, n, n, lda);

            var symmetric = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = symmetric)
            {
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.Trans,
                    n, n, n,
                    1.0,
                    pa, lda,
                    pa, lda,
                    0.0,
                    ps, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    a[lda * col + row] = symmetric[n * col + row];
                }
            }

            return a;
        }

        private static void NanLowerPart(int n, double[] a, int lda)
        {
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (row > col)
                    {
                        a[lda * col + row] = double.NaN;
                    }
                }
            }
        }
    }
}
