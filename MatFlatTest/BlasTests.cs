using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests
    {
        [TestCase(1, 1, 1)]
        [TestCase(1, 2, 3)]
        [TestCase(2, 2, 1)]
        [TestCase(2, 3, 4)]
        [TestCase(3, 3, 1)]
        [TestCase(3, 5, 2)]
        [TestCase(5, 5, 1)]
        [TestCase(5, 7, 3)]
        [TestCase(10, 10, 1)]
        [TestCase(10, 17, 5)]
        public unsafe void ForwardSubstitutionDouble(int n, int lda, int incx)
        {
            var a = Matrix.RandomDouble(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = row + 1; col < n; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomDouble(57, n, incx);

            var expected = input.ToArray();
            fixed (double* pa = a)
            fixed (double* px = expected)
            {
                OpenBlasSharp.Blas.Dtrsv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Uplo.Lower,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Diag.NonUnit,
                    n,
                    pa, lda,
                    px, incx);
            }

            var actual = input.ToArray();
            fixed (double* pa = a)
            fixed (double* px = actual)
            {
                MatFlat.Blas.ForwardSubstitution(n, pa, lda, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
        }
    }
}
