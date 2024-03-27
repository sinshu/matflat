using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_VectorOperations
    {
        [TestCase(1, 1)]
        [TestCase(1, 2)]
        [TestCase(2, 1)]
        [TestCase(2, 3)]
        [TestCase(3, 1)]
        [TestCase(3, 5)]
        [TestCase(10, 1)]
        [TestCase(10, 3)]
        public unsafe void NormSingle(int n, int incx)
        {
            var x = Vector.RandomSingle(42, n, incx);

            float expected;
            fixed (float* px = x)
            {
                expected = OpenBlasSharp.Blas.Snrm2(n, px, incx);
            }

            float actual;
            fixed (float* px = x)
            {
                actual = MatFlat.Blas.L2Norm(n, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
        }

        [TestCase(1, 1)]
        [TestCase(1, 2)]
        [TestCase(2, 1)]
        [TestCase(2, 3)]
        [TestCase(3, 1)]
        [TestCase(3, 5)]
        [TestCase(10, 1)]
        [TestCase(10, 3)]
        public unsafe void NormDouble(int n, int incx)
        {
            var x = Vector.RandomDouble(42, n, incx);

            double expected;
            fixed (double* px = x)
            {
                expected = OpenBlasSharp.Blas.Dnrm2(n, px, incx);
            }

            double actual;
            fixed (double* px = x)
            {
                actual = MatFlat.Blas.L2Norm(n, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
        }

        [TestCase(1, 1)]
        [TestCase(1, 2)]
        [TestCase(2, 1)]
        [TestCase(2, 3)]
        [TestCase(3, 1)]
        [TestCase(3, 5)]
        [TestCase(10, 1)]
        [TestCase(10, 3)]
        public unsafe void NormComplex(int n, int incx)
        {
            var x = Vector.RandomComplex(42, n, incx);

            double expected;
            fixed (Complex* px = x)
            {
                expected = OpenBlasSharp.Blas.Dznrm2(n, px, incx);
            }

            double actual;
            fixed (Complex* px = x)
            {
                actual = MatFlat.Blas.L2Norm(n, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
        }

        [TestCase(1, 1, 1)]
        [TestCase(1, 2, 2)]
        [TestCase(2, 1, 1)]
        [TestCase(2, 3, 2)]
        [TestCase(3, 1, 1)]
        [TestCase(3, 5, 7)]
        [TestCase(10, 1, 1)]
        [TestCase(10, 3, 4)]
        public unsafe void DotSingle(int n, int incx, int incy)
        {
            var x = Vector.RandomSingle(42, n, incx);
            var y = Vector.RandomSingle(57, n, incy);

            float expected;
            fixed (float* px = x)
            fixed (float* py = y)
            {
                expected = OpenBlasSharp.Blas.Sdot(n, px, incx, py, incy);
            }

            float actual;
            fixed (float* px = x)
            fixed (float* py = y)
            {
                actual = MatFlat.Blas.Dot(n, px, incx, py, incy);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
        }

        [TestCase(1, 1, 1)]
        [TestCase(1, 2, 2)]
        [TestCase(2, 1, 1)]
        [TestCase(2, 3, 2)]
        [TestCase(3, 1, 1)]
        [TestCase(3, 5, 7)]
        [TestCase(10, 1, 1)]
        [TestCase(10, 3, 4)]
        public unsafe void DotDouble(int n, int incx, int incy)
        {
            var x = Vector.RandomDouble(42, n, incx);
            var y = Vector.RandomDouble(57, n, incy);

            double expected;
            fixed (double* px = x)
            fixed (double* py = y)
            {
                expected = OpenBlasSharp.Blas.Ddot(n, px, incx, py, incy);
            }

            double actual;
            fixed (double* px = x)
            fixed (double* py = y)
            {
                actual = MatFlat.Blas.Dot(n, px, incx, py, incy);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
        }

        [TestCase(1, 1, 1)]
        [TestCase(1, 2, 2)]
        [TestCase(2, 1, 1)]
        [TestCase(2, 3, 2)]
        [TestCase(3, 1, 1)]
        [TestCase(3, 5, 7)]
        [TestCase(10, 1, 1)]
        [TestCase(10, 3, 4)]
        public unsafe void DotComplex(int n, int incx, int incy)
        {
            var x = Vector.RandomComplex(42, n, incx);
            var y = Vector.RandomComplex(57, n, incy);

            Complex expected;
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            {
                expected = OpenBlasSharp.Blas.Zdotu(n, px, incx, py, incy);
            }

            Complex actual;
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            {
                actual = MatFlat.Blas.Dot(n, px, incx, py, incy);
            }

            Assert.That(actual.Real, Is.EqualTo(expected.Real).Within(1.0E-12));
            Assert.That(actual.Imaginary, Is.EqualTo(expected.Imaginary).Within(1.0E-12));
        }

        [TestCase(1, 1, 1)]
        [TestCase(1, 2, 2)]
        [TestCase(2, 1, 1)]
        [TestCase(2, 3, 2)]
        [TestCase(3, 1, 1)]
        [TestCase(3, 5, 7)]
        [TestCase(10, 1, 1)]
        [TestCase(10, 3, 4)]
        public unsafe void DotConj(int n, int incx, int incy)
        {
            var x = Vector.RandomComplex(42, n, incx);
            var y = Vector.RandomComplex(57, n, incy);

            Complex expected;
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            {
                expected = OpenBlasSharp.Blas.Zdotc(n, px, incx, py, incy);
            }

            Complex actual;
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            {
                actual = MatFlat.Blas.DotConj(n, px, incx, py, incy);
            }

            Assert.That(actual.Real, Is.EqualTo(expected.Real).Within(1.0E-12));
            Assert.That(actual.Imaginary, Is.EqualTo(expected.Imaginary).Within(1.0E-12));
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 4, 3, 2)]
        [TestCase(2, 2, 1, 1, 2)]
        [TestCase(2, 2, 2, 3, 4)]
        [TestCase(3, 3, 1, 1, 3)]
        [TestCase(3, 3, 2, 2, 4)]
        [TestCase(2, 5, 1, 1, 2)]
        [TestCase(2, 5, 3, 2, 4)]
        [TestCase(7, 3, 1, 1, 7)]
        [TestCase(7, 3, 3, 4, 9)]
        public unsafe void OuterSingle(int m, int n, int incx, int incy, int lda)
        {
            var x = Vector.RandomSingle(42, m, incx);
            var y = Vector.RandomSingle(57, n, incy);
            var a = Matrix.RandomSingle(0, m, n, lda);

            var expected = a.ToArray();
            fixed (float* px = x)
            fixed (float* py = y)
            fixed (float* pa = expected)
            {
                for (var j = 0; j < n; j++)
                {
                    new Span<float>(pa + lda * j, m).Clear();
                }
                OpenBlasSharp.Blas.Sger(
                    OpenBlasSharp.Order.ColMajor,
                    m, n,
                    1.0F,
                    px, incx,
                    py, incy,
                    pa, lda);
            }

            var actual = a.ToArray();
            fixed (float* px = x)
            fixed (float* py = y)
            fixed (float* pa = actual)
            {
                MatFlat.Blas.Outer(m, n, px, incx, py, incy, pa, lda);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 4, 3, 2)]
        [TestCase(2, 2, 1, 1, 2)]
        [TestCase(2, 2, 2, 3, 4)]
        [TestCase(3, 3, 1, 1, 3)]
        [TestCase(3, 3, 2, 2, 4)]
        [TestCase(2, 5, 1, 1, 2)]
        [TestCase(2, 5, 3, 2, 4)]
        [TestCase(7, 3, 1, 1, 7)]
        [TestCase(7, 3, 3, 4, 9)]
        public unsafe void OuterDouble(int m, int n, int incx, int incy, int lda)
        {
            var x = Vector.RandomDouble(42, m, incx);
            var y = Vector.RandomDouble(57, n, incy);
            var a = Matrix.RandomDouble(0, m, n, lda);

            var expected = a.ToArray();
            fixed (double* px = x)
            fixed (double* py = y)
            fixed (double* pa = expected)
            {
                for (var j = 0; j < n; j++)
                {
                    new Span<double>(pa + lda * j, m).Clear();
                }
                OpenBlasSharp.Blas.Dger(
                    OpenBlasSharp.Order.ColMajor,
                    m, n,
                    1.0,
                    px, incx,
                    py, incy,
                    pa, lda);
            }

            var actual = a.ToArray();
            fixed (double* px = x)
            fixed (double* py = y)
            fixed (double* pa = actual)
            {
                MatFlat.Blas.Outer(m, n, px, incx, py, incy, pa, lda);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 4, 3, 2)]
        [TestCase(2, 2, 1, 1, 2)]
        [TestCase(2, 2, 2, 3, 4)]
        [TestCase(3, 3, 1, 1, 3)]
        [TestCase(3, 3, 2, 2, 4)]
        [TestCase(2, 5, 1, 1, 2)]
        [TestCase(2, 5, 3, 2, 4)]
        [TestCase(7, 3, 1, 1, 7)]
        [TestCase(7, 3, 3, 4, 9)]
        public unsafe void OuterComplex(int m, int n, int incx, int incy, int lda)
        {
            var x = Vector.RandomComplex(42, m, incx);
            var y = Vector.RandomComplex(57, n, incy);
            var a = Matrix.RandomComplex(0, m, n, lda);

            var expected = a.ToArray();
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            fixed (Complex* pa = expected)
            {
                var one = Complex.One;

                for (var j = 0; j < n; j++)
                {
                    new Span<Complex>(pa + lda * j, m).Clear();
                }
                OpenBlasSharp.Blas.Zgeru(
                    OpenBlasSharp.Order.ColMajor,
                    m, n,
                    &one,
                    px, incx,
                    py, incy,
                    pa, lda);
            }

            var actual = a.ToArray();
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            fixed (Complex* pa = actual)
            {
                MatFlat.Blas.Outer(m, n, px, incx, py, incy, pa, lda);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 4, 3, 2)]
        [TestCase(2, 2, 1, 1, 2)]
        [TestCase(2, 2, 2, 3, 4)]
        [TestCase(3, 3, 1, 1, 3)]
        [TestCase(3, 3, 2, 2, 4)]
        [TestCase(2, 5, 1, 1, 2)]
        [TestCase(2, 5, 3, 2, 4)]
        [TestCase(7, 3, 1, 1, 7)]
        [TestCase(7, 3, 3, 4, 9)]
        public unsafe void OuterConj(int m, int n, int incx, int incy, int lda)
        {
            var x = Vector.RandomComplex(42, m, incx);
            var y = Vector.RandomComplex(57, n, incy);
            var a = Matrix.RandomComplex(0, m, n, lda);

            var expected = a.ToArray();
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            fixed (Complex* pa = expected)
            {
                var one = Complex.One;

                for (var j = 0; j < n; j++)
                {
                    new Span<Complex>(pa + lda * j, m).Clear();
                }
                OpenBlasSharp.Blas.Zgerc(
                    OpenBlasSharp.Order.ColMajor,
                    m, n,
                    &one,
                    px, incx,
                    py, incy,
                    pa, lda);
            }

            var actual = a.ToArray();
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            fixed (Complex* pa = actual)
            {
                MatFlat.Blas.OuterConj(m, n, px, incx, py, incy, pa, lda);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
        }
    }
}
