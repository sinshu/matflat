using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;
using Newtonsoft.Json.Linq;

namespace MatFlatTest
{
    public class CholeskyTests
    {
        [TestCase(1, 1)]
        [TestCase(1, 3)]
        [TestCase(2, 2)]
        [TestCase(2, 3)]
        [TestCase(3, 3)]
        [TestCase(3, 5)]
        [TestCase(11, 11)]
        [TestCase(11, 17)]
        [TestCase(23, 23)]
        [TestCase(23, 31)]
        public unsafe void CholeskyDouble_General(int n, int lda)
        {
            var a = GetDecomposableDouble(42, n, lda);

            var expectedA = a.ToArray();
            fixed (double* pa = expectedA)
            {
                Lapack.Dpotrf(MatrixLayout.ColMajor, 'U', n, pa, lda);
                OpenBlasResultToL(n, expectedA, lda);
            }

            var actualA = a.ToArray();
            fixed (double* pa = actualA)
            {
                Factorization.CholeskyDouble(n, pa, lda);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-12));
        }

        [TestCase(1, 1)]
        [TestCase(1, 3)]
        [TestCase(2, 2)]
        [TestCase(2, 3)]
        [TestCase(3, 3)]
        [TestCase(3, 5)]
        [TestCase(11, 11)]
        [TestCase(11, 17)]
        [TestCase(23, 23)]
        [TestCase(23, 31)]
        public unsafe void CholeskyDouble_MaybeSingular(int n, int lda)
        {
            var a = GetMaybeSingularDouble(42, n, lda);

            bool decomposable;
            var expectedA = a.ToArray();
            fixed (double* pa = expectedA)
            {
                var result = Lapack.Dpotrf(MatrixLayout.ColMajor, 'U', n, pa, lda);
                OpenBlasResultToL(n, expectedA, lda);
                decomposable = result == 0;
            }

            var actualA = a.ToArray();
            fixed (double* pa = actualA)
            {
                try
                {
                    Factorization.CholeskyDouble(n, pa, lda);
                    if (!decomposable)
                    {
                        Assert.Fail();
                    }
                }
                catch
                {
                    if (decomposable)
                    {
                        Assert.Fail();
                    }
                }
            }

            if (decomposable)
            {
                Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-12));
            }
        }

        private static unsafe double[] GetDecomposableDouble(int seed, int n, int lda)
        {
            var a = Matrix.RandomDouble(seed, n, n, lda);

            var symmetric = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = symmetric)
            {
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans,
                    Transpose.Trans,
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
                    if (col >= row)
                    {
                        var value = symmetric[n * col + row];
                        a[lda * col + row] = value;
                    }
                    else
                    {
                        a[lda * col + row] = double.NaN;
                    }
                }
            }

            return a;
        }

        private static unsafe double[] GetMaybeSingularDouble(int seed, int n, int lda)
        {
            var a = Matrix.RandomDouble(seed, n, n, lda);

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (col < row)
                    {
                        a[lda * col + row] = double.NaN;
                    }
                }
            }

            return a;
        }

        private static void OpenBlasResultToL(int n, double[] a, int lda)
        {
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col <= row; col++)
                {
                    var index1 = lda * col + row;
                    var index2 = lda * row + col;
                    if (row != col)
                    {
                        (a[index1], a[index2]) = (a[index2], a[index1]);
                    }
                }
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = row + 1; col < n; col++)
                {
                    var index = lda * col + row;
                    a[index] = 0;
                }
            }
        }
    }
}
