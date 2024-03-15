using System;
using System.Linq;
using System.Numerics;
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
        public unsafe void LuSingle_General(int m, int n, int lda)
        {
            var a = Matrix.RandomSingle(42, m, n, lda);

            var expectedA = a.ToArray();
            var expectedPiv = new int[Math.Min(m, n)];
            fixed (float* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Sgetrf(MatrixLayout.ColMajor, m, n, pa, lda, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[m];
            fixed (float* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuSingle(m, n, pa, lda, ppiv);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-6));
        }

        [Test]
        public unsafe void LuSingle_Zero()
        {
            var a = new float[9];

            var expectedA = a.ToArray();
            var expectedPiv = new int[3];
            fixed (float* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Sgetrf(MatrixLayout.ColMajor, 3, 3, pa, 3, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[3];
            fixed (float* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuSingle(3, 3, pa, 3, ppiv);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-6));
        }

        [Test]
        public unsafe void LuSingle_Singular()
        {
            var a = Enumerable.Range(0, 9).Select(i => (float)(i + 1)).ToArray();

            var expectedA = a.ToArray();
            var expectedPiv = new int[3];
            fixed (float* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Sgetrf(MatrixLayout.ColMajor, 3, 3, pa, 3, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[3];
            fixed (float* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuSingle(3, 3, pa, 3, ppiv);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-6));
        }

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
        public unsafe void LuDouble_General(int m, int n, int lda)
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
                Factorization.LuDouble(m, n, pa, lda, ppiv);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-12));
        }

        [Test]
        public unsafe void LuDouble_Zero()
        {
            var a = new double[9];

            var expectedA = a.ToArray();
            var expectedPiv = new int[3];
            fixed (double* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Dgetrf(MatrixLayout.ColMajor, 3, 3, pa, 3, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[3];
            fixed (double* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuDouble(3, 3, pa, 3, ppiv);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-12));
        }

        [Test]
        public unsafe void LuDouble_Singular()
        {
            var a = Enumerable.Range(0, 9).Select(i => (double)(i + 1)).ToArray();

            var expectedA = a.ToArray();
            var expectedPiv = new int[3];
            fixed (double* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Dgetrf(MatrixLayout.ColMajor, 3, 3, pa, 3, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[3];
            fixed (double* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuDouble(3, 3, pa, 3, ppiv);
            }

            Assert.That(actualA, Is.EqualTo(expectedA).Within(1.0E-12));
        }

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
        public unsafe void LuComplex_General(int m, int n, int lda)
        {
            var a = Matrix.RandomComplex(42, m, n, lda);

            var expectedA = a.ToArray();
            var expectedPiv = new int[Math.Min(m, n)];
            fixed (Complex* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Zgetrf(MatrixLayout.ColMajor, m, n, pa, lda, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[m];
            fixed (Complex* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuComplex(m, n, pa, lda, ppiv);
            }

            Assert.That(actualA.Select(x => x.Real), Is.EqualTo(expectedA.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actualA.Select(x => x.Imaginary), Is.EqualTo(expectedA.Select(x => x.Imaginary)).Within(1.0E-12));
        }

        [Test]
        public unsafe void LuComplex_Zero()
        {
            var a = new Complex[9];

            var expectedA = a.ToArray();
            var expectedPiv = new int[3];
            fixed (Complex* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Zgetrf(MatrixLayout.ColMajor, 3, 3, pa, 3, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[3];
            fixed (Complex* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuComplex(3, 3, pa, 3, ppiv);
            }

            Assert.That(actualA.Select(x => x.Real), Is.EqualTo(expectedA.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actualA.Select(x => x.Imaginary), Is.EqualTo(expectedA.Select(x => x.Imaginary)).Within(1.0E-12));
        }

        [Test]
        public unsafe void LuComplex_Singular()
        {
            var a = Enumerable.Range(0, 9).Select(i => (Complex)(i + 1)).ToArray();

            var expectedA = a.ToArray();
            var expectedPiv = new int[3];
            fixed (Complex* pa = expectedA)
            fixed (int* ppiv = expectedPiv)
            {
                Lapack.Zgetrf(MatrixLayout.ColMajor, 3, 3, pa, 3, ppiv);
            }

            var actualA = a.ToArray();
            var actualPiv = new int[3];
            fixed (Complex* pa = actualA)
            fixed (int* ppiv = actualPiv)
            {
                Factorization.LuComplex(3, 3, pa, 3, ppiv);
            }

            Assert.That(actualA.Select(x => x.Real), Is.EqualTo(expectedA.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actualA.Select(x => x.Imaginary), Is.EqualTo(expectedA.Select(x => x.Imaginary)).Within(1.0E-12));
        }
    }
}
