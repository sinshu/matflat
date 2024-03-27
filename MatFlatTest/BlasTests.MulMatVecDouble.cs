using System;
using System.Linq;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_MulMatVecDouble
    {
        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 1, 1)]
        [TestCase(2, 2, 4, 3, 3)]
        [TestCase(3, 3, 3, 1, 1)]
        [TestCase(3, 3, 5, 6, 7)]
        [TestCase(1, 5, 1, 1, 1)]
        [TestCase(1, 5, 3, 2, 4)]
        [TestCase(4, 1, 4, 1, 1)]
        [TestCase(4, 1, 6, 5, 7)]
        [TestCase(4, 9, 4, 1, 1)]
        [TestCase(4, 9, 5, 3, 2)]
        [TestCase(8, 3, 8, 1, 1)]
        [TestCase(8, 3, 9, 2, 2)]
        public unsafe void NoTrans(int m, int n, int lda, int incx, int incy)
        {
            var a = Matrix.RandomDouble(42, m, n, lda);
            var x = Vector.RandomDouble(57, n, incx);
            var y = Vector.RandomDouble(0, m, incy);

            var expected = y.ToArray();
            fixed (double* pa = a)
            fixed (double* px = x)
            fixed (double* py = expected)
            {
                OpenBlasSharp.Blas.Dgemv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    m, n,
                    1.0,
                    pa, lda,
                    px, incx,
                    0.0,
                    py, incy);
            }

            var actual = y.ToArray();
            fixed (double* pa = a)
            fixed (double* px = x)
            fixed (double* py = actual)
            {
                MatFlat.Blas.MulMatVec(MatFlat.Transpose.NoTrans, m, n, pa, lda, px, incx, py, incy);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 1, 1)]
        [TestCase(2, 2, 3, 2, 3)]
        [TestCase(3, 3, 3, 1, 1)]
        [TestCase(3, 3, 4, 3, 2)]
        [TestCase(1, 5, 1, 1, 1)]
        [TestCase(1, 5, 2, 2, 2)]
        [TestCase(4, 1, 4, 1, 1)]
        [TestCase(4, 1, 5, 3, 4)]
        [TestCase(4, 9, 4, 1, 1)]
        [TestCase(4, 9, 5, 3, 2)]
        [TestCase(8, 3, 8, 1, 1)]
        [TestCase(8, 3, 9, 2, 2)]
        public unsafe void Trans(int m, int n, int lda, int incx, int incy)
        {
            var a = Matrix.RandomDouble(42, m, n, lda);
            var x = Vector.RandomDouble(57, n, incx);
            var y = Vector.RandomDouble(0, n, incy);

            var expected = y.ToArray();
            fixed (double* pa = a)
            fixed (double* px = x)
            fixed (double* py = expected)
            {
                OpenBlasSharp.Blas.Dgemv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans,
                    m, n,
                    1.0,
                    pa, lda,
                    px, incx,
                    0.0,
                    py, incy);
            }

            var actual = y.ToArray();
            fixed (double* pa = a)
            fixed (double* px = x)
            fixed (double* py = actual)
            {
                MatFlat.Blas.MulMatVec(MatFlat.Transpose.Trans, m, n, pa, lda, px, incx, py, incy);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
        }
    }
}
