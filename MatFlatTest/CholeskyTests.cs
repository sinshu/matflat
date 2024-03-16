using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

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
        public unsafe void CholeskyComplex_General(int n, int lda)
        {
            var a = GetDecomposableComplex(42, n, lda);

            var expectedA = a.ToArray();
            fixed (Complex* pa = expectedA)
            {
                Lapack.Zpotrf(MatrixLayout.ColMajor, 'U', n, pa, lda);
                OpenBlasResultToL(n, expectedA, lda);
            }

            var actualA = a.ToArray();
            fixed (Complex* pa = actualA)
            {
                Factorization.CholeskyComplex(n, pa, lda);
            }

            Assert.That(actualA.Select(x => x.Real), Is.EqualTo(expectedA.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actualA.Select(x => x.Imaginary), Is.EqualTo(expectedA.Select(x => x.Imaginary)).Within(1.0E-12));
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
        public unsafe void CholeskyComplex_MaybeSingular(int n, int lda)
        {
            var a = GetMaybeSingularComplex(42, n, lda);

            bool decomposable;
            var expectedA = a.ToArray();
            fixed (Complex* pa = expectedA)
            {
                var result = Lapack.Zpotrf(MatrixLayout.ColMajor, 'U', n, pa, lda);
                OpenBlasResultToL(n, expectedA, lda);
                decomposable = result == 0;
            }

            var actualA = a.ToArray();
            fixed (Complex* pa = actualA)
            {
                try
                {
                    Factorization.CholeskyComplex(n, pa, lda);
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
                Assert.That(actualA.Select(x => x.Real), Is.EqualTo(expectedA.Select(x => x.Real)).Within(1.0E-12));
                Assert.That(actualA.Select(x => x.Imaginary), Is.EqualTo(expectedA.Select(x => x.Imaginary)).Within(1.0E-12));
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

        private static unsafe Complex[] GetDecomposableComplex(int seed, int n, int lda)
        {
            var a = Matrix.RandomComplex(seed, n, n, lda);

            var symmetric = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (Complex* ps = symmetric)
            {
                var one = Complex.One;
                var zero = Complex.Zero;
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans,
                    Transpose.ConjTrans,
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
                    if (col >= row)
                    {
                        var value = symmetric[n * col + row];
                        a[lda * col + row] = value;
                    }
                    else
                    {
                        a[lda * col + row] = Complex.NaN;
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

        private static unsafe Complex[] GetMaybeSingularComplex(int seed, int n, int lda)
        {
            var a = Matrix.RandomComplex(seed, n, n, lda);

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (col < row)
                    {
                        a[lda * col + row] = Complex.NaN;
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

        private static void OpenBlasResultToL(int n, Complex[] a, int lda)
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
                for (var col = 0; col < n; col++)
                {
                    var index = lda * col + row;
                    if (col > row)
                    {
                        a[index] = 0;
                    }
                    else
                    {
                        var x = a[index];
                        a[index] = new Complex(x.Real, -x.Imaginary);
                    }
                }
            }
        }
    }
}
