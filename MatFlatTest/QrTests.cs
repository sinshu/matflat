using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class QrTests
    {
        [TestCase(1, 1, 1)]
        [TestCase(1, 1, 3)]
        [TestCase(2, 2, 2)]
        [TestCase(2, 2, 3)]
        [TestCase(3, 3, 3)]
        [TestCase(3, 3, 5)]
        [TestCase(3, 1, 3)]
        [TestCase(3, 1, 5)]
        [TestCase(4, 3, 4)]
        [TestCase(4, 3, 5)]
        [TestCase(16, 8, 16)]
        [TestCase(16, 8, 32)]
        [TestCase(23, 11, 23)]
        [TestCase(23, 11, 31)]
        public unsafe void QrDouble_General(int m, int n, int lda)
        {
            var a = Matrix.RandomDouble(42, m, n, lda);

            var expectedA = a.ToArray();
            var expectedTau = new double[n];
            fixed (double* pa = expectedA)
            fixed (double* ptau = expectedTau)
            {
                Lapack.Dgeqrf(MatrixLayout.ColMajor, m, n, pa, lda, ptau);
            }

            var actualA = a.ToArray();
            var actualRdiag = new double[n];
            fixed (double* pa = actualA)
            fixed (double* prdiag = actualRdiag)
            {
                Factorization.QrDouble(m, n, pa, lda, prdiag);
            }

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (col > row)
                {
                    Assert.That(actualA[i], Is.EqualTo(expectedA[i]).Within(1.0E-12));
                }
                else if (row == col)
                {
                    Assert.That(Math.Abs(actualRdiag[col]), Is.EqualTo(Math.Abs(expectedA[i])).Within(1.0E-12));
                }
                else if (row >= lda)
                {
                    Assert.That(actualA[i], Is.EqualTo(expectedA[i]).Within(1.0E-12));
                }
            }
        }
    }
}
