using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class EvdTests
    {
        [TestCase(1, 1)]
        [TestCase(1, 3)]
        [TestCase(2, 2)]
        [TestCase(2, 3)]
        [TestCase(3, 3)]
        [TestCase(3, 5)]
        [TestCase(5, 5)]
        [TestCase(5, 8)]
        [TestCase(10, 10)]
        [TestCase(10, 17)]
        public unsafe void EvdSingle(int n, int lda)
        {
            var a = GetDecomposableSingle(42, n, lda);

            var v = a.ToArray();
            var w = new float[n];

            NanLowerPart(n, v, lda);

            fixed (float* pv = v)
            fixed (float* pw = w)
            {
                Factorization.Evd(n, pv, lda, pw);
            }

            var diag = new float[n * n];
            for (var i = 0; i < n; i++)
            {
                diag[n * i + i] = w[i];
            }

            var tmp = new float[n * n];
            var reconstructed = new float[n * n];

            fixed (float* pa = a)
            fixed (float* pv = v)
            fixed (float* pdiag = diag)
            fixed (float* ptmp = tmp)
            fixed (float* preconstructed = reconstructed)
            {
                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.NoTrans,
                    n, n, n,
                    1.0F,
                    pv, lda,
                    pdiag, n,
                    0.0F,
                    ptmp, n);

                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.Trans,
                    n, n, n,
                    1.0F,
                    ptmp, n,
                    pv, lda,
                    0.0F,
                    preconstructed, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(n, n, reconstructed, n, row, col);
                    var expected = Matrix.Get(n, n, a, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-5));
                }
            }

            for (var i = 0; i < v.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= n)
                {
                    Assert.That(v[i], Is.EqualTo(a[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1)]
        [TestCase(1, 3)]
        [TestCase(2, 2)]
        [TestCase(2, 3)]
        [TestCase(3, 3)]
        [TestCase(3, 5)]
        [TestCase(5, 5)]
        [TestCase(5, 8)]
        [TestCase(10, 10)]
        [TestCase(10, 17)]
        public unsafe void EvdDouble(int n, int lda)
        {
            var a = GetDecomposableDouble(42, n, lda);

            var v = a.ToArray();
            var w = new double[n];

            NanLowerPart(n, v, lda);

            fixed (double* pv = v)
            fixed (double* pw = w)
            {
                Factorization.Evd(n, pv, lda, pw);
            }

            var diag = new double[n * n];
            for (var i = 0; i < n; i++)
            {
                diag[n * i + i] = w[i];
            }

            var tmp = new double[n * n];
            var reconstructed = new double[n * n];

            fixed (double* pa = a)
            fixed (double* pv = v)
            fixed (double* pdiag = diag)
            fixed (double* ptmp = tmp)
            fixed (double* preconstructed = reconstructed)
            {
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.NoTrans,
                    n, n, n,
                    1.0,
                    pv, lda,
                    pdiag, n,
                    0.0,
                    ptmp, n);

                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.Trans,
                    n, n, n,
                    1.0,
                    ptmp, n,
                    pv, lda,
                    0.0,
                    preconstructed, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(n, n, reconstructed, n, row, col);
                    var expected = Matrix.Get(n, n, a, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
                }
            }

            for (var i = 0; i < v.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= n)
                {
                    Assert.That(v[i], Is.EqualTo(a[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1)]
        [TestCase(1, 3)]
        [TestCase(2, 2)]
        [TestCase(2, 3)]
        [TestCase(3, 3)]
        [TestCase(3, 5)]
        [TestCase(5, 5)]
        [TestCase(5, 8)]
        [TestCase(10, 10)]
        [TestCase(10, 17)]
        public unsafe void EvdComplex(int n, int lda)
        {
            var a = GetDecomposableComplex(42, n, lda);

            var v = a.ToArray();
            var w = new double[n];

            NanLowerPart(n, v, lda);

            fixed (Complex* pv = v)
            fixed (double* pw = w)
            {
                Factorization.Evd(n, pv, lda, pw);
            }

            var diag = new Complex[n * n];
            for (var i = 0; i < n; i++)
            {
                diag[n * i + i] = w[i];
            }

            var tmp = new Complex[n * n];
            var reconstructed = new Complex[n * n];

            fixed (Complex* pa = a)
            fixed (Complex* pv = v)
            fixed (Complex* pdiag = diag)
            fixed (Complex* ptmp = tmp)
            fixed (Complex* preconstructed = reconstructed)
            {
                var one = Complex.One;
                var zero = Complex.Zero;

                OpenBlasSharp.Blas.Zgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.NoTrans,
                    n, n, n,
                    &one,
                    pv, lda,
                    pdiag, n,
                    &zero,
                    ptmp, n);

                OpenBlasSharp.Blas.Zgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.ConjTrans,
                    n, n, n,
                    &one,
                    ptmp, n,
                    pv, lda,
                    &zero,
                    preconstructed, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(n, n, reconstructed, n, row, col);
                    var expected = Matrix.Get(n, n, a, lda, row, col);
                    Assert.That(actual.Real, Is.EqualTo(expected.Real).Within(1.0E-12));
                    Assert.That(actual.Imaginary, Is.EqualTo(expected.Imaginary).Within(1.0E-12));
                }
            }

            for (var i = 0; i < v.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= n)
                {
                    Assert.That(v[i].Real, Is.EqualTo(a[i].Real).Within(0));
                    Assert.That(v[i].Imaginary, Is.EqualTo(a[i].Imaginary).Within(0));
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
