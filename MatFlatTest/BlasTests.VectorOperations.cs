using System;
using System.Linq;
using System.Numerics;
using MatFlat;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_VectorOperations
    {
        private static unsafe float FakeSnrm2(int n, float* x, int incx)
        {
            return MathF.Sqrt(FakeSdot(n, x, incx, x, incx));
        }

        private static unsafe double FakeDnrm2(int n, double* x, int incx)
        {
            return Math.Sqrt(FakeDdot(n, x, incx, x, incx));
        }

        private static unsafe double FakeDznrm2(int n, Complex* x, int incx)
        {
            return Math.Sqrt(FakeZdotc(n, x, incx, x, incx).Real);
        }

        private static unsafe float FakeSdot(int n, float* x, int incx, float* y, int incy)
        {
            var sum = 0.0F;
            for (var i = 0; i < n; i++)
            {
                sum += x[i * incx] * y[i * incy];
            }

            return sum;
        }

        private static unsafe double FakeDdot(int n, double* x, int incx, double* y, int incy)
        {
            var sum = 0.0;
            for (var i = 0; i < n; i++)
            {
                sum += x[i * incx] * y[i * incy];
            }

            return sum;
        }

        private static unsafe Complex FakeZdotu(int n, Complex* x, int incx, Complex* y, int incy)
        {
            var sum = Complex.Zero;
            for (var i = 0; i < n; i++)
            {
                sum += x[i * incx] * y[i * incy];
            }

            return sum;
        }

        private static unsafe Complex FakeZdotc(int n, Complex* x, int incx, Complex* y, int incy)
        {
            var sum = Complex.Zero;
            for (var i = 0; i < n; i++)
            {
                sum += Complex.Conjugate(x[i * incx]) * y[i * incy];
            }

            return sum;
        }

        private static unsafe void FakeSger(int m, int n, float* x, int incx, float* y, int incy, float* a, int lda)
        {
            for (var col = 0; col < n; col++)
            {
                for (var row = 0; row < m; row++)
                {
                    a[(col * lda) + row] = x[row * incx] * y[col * incy];
                }
            }
        }

        private static unsafe void FakeDger(int m, int n, double* x, int incx, double* y, int incy, double* a, int lda)
        {
            for (var col = 0; col < n; col++)
            {
                for (var row = 0; row < m; row++)
                {
                    a[(col * lda) + row] = x[row * incx] * y[col * incy];
                }
            }
        }

        private static unsafe void FakeZgeru(int m, int n, Complex* x, int incx, Complex* y, int incy, Complex* a, int lda)
        {
            for (var col = 0; col < n; col++)
            {
                for (var row = 0; row < m; row++)
                {
                    a[(col * lda) + row] = x[row * incx] * y[col * incy];
                }
            }
        }

        private static unsafe void FakeZgerc(int m, int n, Complex* x, int incx, Complex* y, int incy, Complex* a, int lda)
        {
            for (var col = 0; col < n; col++)
            {
                for (var row = 0; row < m; row++)
                {
                    a[(col * lda) + row] = x[row * incx] * Complex.Conjugate(y[col * incy]);
                }
            }
        }

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
                expected = FakeSnrm2(n, px, incx);
            }

            float actual;
            fixed (float* px = x)
            {
                actual = Blas.L2Norm(n, px, incx);
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
                expected = FakeDnrm2(n, px, incx);
            }

            double actual;
            fixed (double* px = x)
            {
                actual = Blas.L2Norm(n, px, incx);
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
                expected = FakeDznrm2(n, px, incx);
            }

            double actual;
            fixed (Complex* px = x)
            {
                actual = Blas.L2Norm(n, px, incx);
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
                expected = FakeSdot(n, px, incx, py, incy);
            }

            float actual;
            fixed (float* px = x)
            fixed (float* py = y)
            {
                actual = Blas.Dot(n, px, incx, py, incy);
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
                expected = FakeDdot(n, px, incx, py, incy);
            }

            double actual;
            fixed (double* px = x)
            fixed (double* py = y)
            {
                actual = Blas.Dot(n, px, incx, py, incy);
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
                expected = FakeZdotu(n, px, incx, py, incy);
            }

            Complex actual;
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            {
                actual = Blas.Dot(n, px, incx, py, incy);
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
                expected = FakeZdotc(n, px, incx, py, incy);
            }

            Complex actual;
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            {
                actual = Blas.DotConj(n, px, incx, py, incy);
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
                FakeSger(m, n, px, incx, py, incy, pa, lda);
            }

            var actual = a.ToArray();
            fixed (float* px = x)
            fixed (float* py = y)
            fixed (float* pa = actual)
            {
                Blas.Outer(m, n, px, incx, py, incy, pa, lda);
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
                FakeDger(m, n, px, incx, py, incy, pa, lda);
            }

            var actual = a.ToArray();
            fixed (double* px = x)
            fixed (double* py = y)
            fixed (double* pa = actual)
            {
                Blas.Outer(m, n, px, incx, py, incy, pa, lda);
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
                FakeZgeru(m, n, px, incx, py, incy, pa, lda);
            }

            var actual = a.ToArray();
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            fixed (Complex* pa = actual)
            {
                Blas.Outer(m, n, px, incx, py, incy, pa, lda);
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
                FakeZgerc(m, n, px, incx, py, incy, pa, lda);
            }

            var actual = a.ToArray();
            fixed (Complex* px = x)
            fixed (Complex* py = y)
            fixed (Complex* pa = actual)
            {
                Blas.OuterConj(m, n, px, incx, py, incy, pa, lda);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
        }
    }
}
