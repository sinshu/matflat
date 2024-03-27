using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_MulMatVecComplex
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
            var a = Matrix.RandomComplex(42, m, n, lda);
            var x = Vector.RandomComplex(57, n, incx);
            var y = Vector.RandomComplex(0, m, incy);

            var expected = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = expected)
            {
                var one = Complex.One;
                var zero = Complex.Zero;

                OpenBlasSharp.Blas.Zgemv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    m, n,
                    &one,
                    pa, lda,
                    px, incx,
                    &zero,
                    py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                MatFlat.Blas.MulMatVec(MatFlat.Transpose.NoTrans, m, n, pa, lda, px, incx, py, incy);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
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
        [TestCase(5, 9, 5, 1, 1)]
        [TestCase(5, 9, 6, 3, 2)]
        [TestCase(8, 3, 8, 1, 1)]
        [TestCase(8, 3, 9, 2, 2)]
        public unsafe void Trans(int m, int n, int lda, int incx, int incy)
        {
            var a = Matrix.RandomComplex(42, m, n, lda);
            var x = Vector.RandomComplex(57, n, incx);
            var y = Vector.RandomComplex(0, n, incy);

            var expected = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = expected)
            {
                var one = Complex.One;
                var zero = Complex.Zero;

                OpenBlasSharp.Blas.Zgemv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans,
                    m, n,
                    &one,
                    pa, lda,
                    px, incx,
                    &zero,
                    py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                MatFlat.Blas.MulMatVec(MatFlat.Transpose.Trans, m, n, pa, lda, px, incx, py, incy);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
        }

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
        public unsafe void ConjNoTrans(int m, int n, int lda, int incx, int incy)
        {
            var a = Matrix.RandomComplex(42, m, n, lda);
            var x = Vector.RandomComplex(57, n, incx);
            var y = Vector.RandomComplex(0, m, incy);

            var expected = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = expected)
            {
                var one = Complex.One;
                var zero = Complex.Zero;

                OpenBlasSharp.Blas.Zgemv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.ConjNoTrans,
                    m, n,
                    &one,
                    pa, lda,
                    px, incx,
                    &zero,
                    py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                MatFlat.Blas.MulMatVec(MatFlat.Transpose.ConjNoTrans, m, n, pa, lda, px, incx, py, incy);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
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
        [TestCase(5, 9, 5, 1, 1)]
        [TestCase(5, 9, 6, 3, 2)]
        [TestCase(8, 3, 8, 1, 1)]
        [TestCase(8, 3, 9, 2, 2)]
        public unsafe void ConjTrans(int m, int n, int lda, int incx, int incy)
        {
            var a = Matrix.RandomComplex(42, m, n, lda);
            var x = Vector.RandomComplex(57, n, incx);
            var y = Vector.RandomComplex(0, n, incy);

            var expected = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = expected)
            {
                var one = Complex.One;
                var zero = Complex.Zero;

                OpenBlasSharp.Blas.Zgemv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.ConjTrans,
                    m, n,
                    &one,
                    pa, lda,
                    px, incx,
                    &zero,
                    py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                MatFlat.Blas.MulMatVec(MatFlat.Transpose.ConjTrans, m, n, pa, lda, px, incx, py, incy);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
        }
    }
}
