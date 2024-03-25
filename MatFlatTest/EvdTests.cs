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
        [TestCase(3, 3)]
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

            Assert.That(a, Is.EqualTo(reconstructed).Within(1.0E-12));
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
