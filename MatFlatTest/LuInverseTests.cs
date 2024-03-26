using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class LuInverseTests
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
        public unsafe void LuInverseSingle(int n, int lda)
        {
            var original = Matrix.RandomSingle(42, n, n, lda);

            var a = original.ToArray();
            var piv = new int[n];
            var identity = new float[n * n];
            fixed (float* pa = a)
            fixed (int* ppiv = piv)
            fixed (float* poriginal = original)
            fixed (float* pidentity = identity)
            {
                Factorization.Lu(n, n, pa, lda, ppiv);
                Factorization.LuInverse(n, pa, lda, ppiv);
                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.NoTrans,
                    n, n, n,
                    1.0F,
                    poriginal, lda,
                    pa, lda,
                    0.0F,
                    pidentity, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value, Is.EqualTo(1.0).Within(1.0E-5));
                    }
                    else
                    {
                        Assert.That(value, Is.EqualTo(0.0).Within(1.0E-5));
                    }
                }
            }

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= n)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
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
        public unsafe void LuInverseDouble(int n, int lda)
        {
            var original = Matrix.RandomDouble(42, n, n, lda);

            var a = original.ToArray();
            var piv = new int[n];
            var identity = new double[n * n];
            fixed (double* pa = a)
            fixed (int* ppiv = piv)
            fixed (double* poriginal = original)
            fixed (double* pidentity = identity)
            {
                Factorization.Lu(n, n, pa, lda, ppiv);
                Factorization.LuInverse(n, pa, lda, ppiv);
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.NoTrans,
                    n, n, n,
                    1.0,
                    poriginal, lda,
                    pa, lda,
                    0.0,
                    pidentity, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value, Is.EqualTo(1.0).Within(1.0E-12));
                    }
                    else
                    {
                        Assert.That(value, Is.EqualTo(0.0).Within(1.0E-12));
                    }
                }
            }

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= n)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
                }
            }
        }
    }
}
