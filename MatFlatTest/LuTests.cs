using System;
using System.Linq;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class Tests
    {
        [TestCase(1, 1, 1)]
        [TestCase(1, 1, 3)]
        [TestCase(2, 2, 2)]
        [TestCase(2, 2, 3)]
        [TestCase(3, 3, 3)]
        [TestCase(3, 3, 5)]
        [TestCase(3, 1, 3)]
        [TestCase(3, 1, 5)]
        [TestCase(1, 3, 1)]
        [TestCase(1, 3, 3)]
        [TestCase(3, 4, 3)]
        [TestCase(3, 4, 5)]
        [TestCase(4, 3, 4)]
        [TestCase(4, 3, 5)]
        [TestCase(11, 23, 11)]
        [TestCase(11, 23, 17)]
        [TestCase(23, 11, 23)]
        [TestCase(23, 11, 31)]
        public unsafe void LuDouble(int m, int n, int lda)
        {
            var a = Matrix.RandomDouble(42, m, n, lda);

            var expectedA = a.ToArray();
            var expectedPiv = new int[Math.Min(m, n)];
            fixed (double* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Dgetrf(MatrixLayout.ColMajor, m, n, pa, lda, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[m];
            fixed (double* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                MatrixDecomposition.LuDouble(m, n, pa, lda, ppiv);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-12));
        }
    }
}
