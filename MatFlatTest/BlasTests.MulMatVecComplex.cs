using System;
using System.Linq;
using System.Numerics;
using MatFlat;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_MulMatVecComplex
    {
        private static unsafe void FakeZgemv(char transA, int m, int n, Complex* a, int lda, Complex* x, int incx, Complex* y, int incy)
        {
            if (transA == 'N' || transA == '!')
            {
                for (var row = 0; row < m; row++)
                {
                    var sum = Complex.Zero;
                    for (var col = 0; col < n; col++)
                    {
                        var value = a[(col * lda) + row];
                        if (transA == '!')
                        {
                            value = Complex.Conjugate(value);
                        }

                        sum += value * x[col * incx];
                    }

                    y[row * incy] = sum;
                }
            }
            else
            {
                for (var col = 0; col < n; col++)
                {
                    var sum = Complex.Zero;
                    for (var row = 0; row < m; row++)
                    {
                        var value = a[(col * lda) + row];
                        if (transA == 'C')
                        {
                            value = Complex.Conjugate(value);
                        }

                        sum += value * x[row * incx];
                    }

                    y[col * incy] = sum;
                }
            }
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
                FakeZgemv('N', m, n, pa, lda, px, incx, py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                Blas.MulMatVec(Transpose.NoTrans, m, n, pa, lda, px, incx, py, incy);
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
            var x = Vector.RandomComplex(57, m, incx);
            var y = Vector.RandomComplex(0, n, incy);

            var expected = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = expected)
            {
                FakeZgemv('T', m, n, pa, lda, px, incx, py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                Blas.MulMatVec(Transpose.Trans, m, n, pa, lda, px, incx, py, incy);
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
                FakeZgemv('!', m, n, pa, lda, px, incx, py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                Blas.MulMatVec(Transpose.ConjNoTrans, m, n, pa, lda, px, incx, py, incy);
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
            var x = Vector.RandomComplex(57, m, incx);
            var y = Vector.RandomComplex(0, n, incy);

            var expected = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = expected)
            {
                FakeZgemv('C', m, n, pa, lda, px, incx, py, incy);
            }

            var actual = y.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = x)
            fixed (Complex* py = actual)
            {
                Blas.MulMatVec(Transpose.ConjTrans, m, n, pa, lda, px, incx, py, incy);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
        }
    }
}
