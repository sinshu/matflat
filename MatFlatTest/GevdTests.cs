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
        public unsafe void GevdSingle(int n, int lda, int ldb)
        {
            var a = GetDecomposableSingle(42, n, lda);
            var b = GetDecomposableSingle(57, n, ldb);

            var v = a.ToArray();
            var l = b.ToArray();
            var w = new float[n];

            NanLowerPart(n, v, lda);
            NanLowerPart(n, l, ldb);

            fixed (float* pv = v)
            fixed (float* pl = l)
            fixed (float* pw = w)
            {
                Factorization.Gevd(n, pv, lda, pl, ldb, pw);
            }

            var left = new float[n];
            var right = new float[n];
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pv = v)
            fixed (float* pl = left)
            fixed (float* pr = right)
            {
                for (var j = 0; j < n; j++)
                {
                    OpenBlasSharp.Blas.Sgemv(
                        Order.ColMajor,
                        OpenBlasSharp.Transpose.NoTrans,
                        n, n,
                        1.0F,
                        pa, lda,
                        pv + lda * j, 1,
                        0.0F,
                        pl, 1);

                    OpenBlasSharp.Blas.Sgemv(
                        Order.ColMajor,
                        OpenBlasSharp.Transpose.NoTrans,
                        n, n,
                        w[j],
                        pb, ldb,
                        pv + lda * j, 1,
                        0.0F,
                        pr, 1);

                    Assert.That(left, Is.EqualTo(right).Within(1.0E-3));
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
        public unsafe void GevdComplex(int n, int lda, int ldb)
        {
            var a = GetDecomposableComplex(42, n, lda);
            var b = GetDecomposableComplex(57, n, ldb);

            var v = a.ToArray();
            var l = b.ToArray();
            var w = new double[n];

            NanLowerPart(n, v, lda);
            NanLowerPart(n, l, ldb);

            fixed (Complex* pv = v)
            fixed (Complex* pl = l)
            fixed (double* pw = w)
            {
                Factorization.Gevd(n, pv, lda, pl, ldb, pw);
            }

            var left = new Complex[n];
            var right = new Complex[n];
            fixed (Complex* pa = a)
            fixed (Complex* pb = b)
            fixed (Complex* pv = v)
            fixed (Complex* pl = left)
            fixed (Complex* pr = right)
            {
                var one = Complex.One;
                var zero = Complex.Zero;
                for (var j = 0; j < n; j++)
                {
                    var cw = (Complex)w[j];

                    OpenBlasSharp.Blas.Zgemv(
                        Order.ColMajor,
                        OpenBlasSharp.Transpose.NoTrans,
                        n, n,
                        &one,
                        pa, lda,
                        pv + lda * j, 1,
                        &zero,
                        pl, 1);

                    OpenBlasSharp.Blas.Zgemv(
                        Order.ColMajor,
                        OpenBlasSharp.Transpose.NoTrans,
                        n, n,
                        &cw,
                        pb, ldb,
                        pv + lda * j, 1,
                        &zero,
                        pr, 1);

                    Assert.That(left.Select(x => x.Real), Is.EqualTo(right.Select(x => x.Real)).Within(1.0E-11));
                    Assert.That(left.Select(x => x.Imaginary), Is.EqualTo(right.Select(x => x.Imaginary)).Within(1.0E-11));
                }
            }
        }

        private static unsafe float[] GetDecomposableSingle(int seed, int n, int lda)
        {
            var a = Matrix.RandomSingle(seed, n, n, lda);

            var symmetric = new float[n * n];
            fixed (float* pa = a)
            fixed (float* ps = symmetric)
            {
                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.Trans,
                    n, n, n,
                    1.0F,
                    pa, lda,
                    pa, lda,
                    0.0F,
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

        private static unsafe Complex[] GetDecomposableComplex(int seed, int n, int lda)
        {
            var a = Matrix.RandomComplex(seed, n, n, lda);

            var symmetric = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (Complex* ps = symmetric)
            {
                var one = Complex.One;
                var zero = Complex.Zero;
                OpenBlasSharp.Blas.Zgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.ConjTrans,
                    n, n, n,
                    &one,
                    pa, lda,
                    pa, lda,
                    &zero,
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

        private static void NanLowerPart(int n, float[] a, int lda)
        {
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (row > col)
                    {
                        a[lda * col + row] = float.NaN;
                    }
                }
            }
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

        private static void NanLowerPart(int n, Complex[] a, int lda)
        {
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (row > col)
                    {
                        a[lda * col + row] = Complex.NaN;
                    }
                }
            }
        }
    }
}
