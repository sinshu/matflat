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
            var a = GetTestMatrix(42, n, lda);

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

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-6));
        }

        private static double[] GetTestMatrix(int seed, int n, int lda)
        {
            var a = Matrix.RandomDouble(seed, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col <= row; col++)
                {
                    var index = lda * col + row;
                    if (row == col)
                    {
                        a[index] += n;
                    }
                    else if (col < row)
                    {
                        a[index] = double.NaN;
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
