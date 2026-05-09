using System;
using System.Linq;
using System.Numerics;
using ILNumerics;
using ILNumerics.Core.Native;
using ILNumerics.F2NET;
using MatFlat;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_SolveTriangularComplex
    {
        private static readonly ILapack lapack = new ManagedLAPACK();

        private static unsafe void FakeZtrsv(char uplo, char transA, int n, Complex* a, int lda, Complex* x, int incx)
        {
            // ManagedLAPACK does not support OpenBLAS' ConjNoTrans mode, so this
            // test helper treats '!' as "conjugate without transpose".
            var aCopy = new Complex[lda * n];
            for (var i = 0; i < aCopy.Length; i++)
            {
                aCopy[i] = a[i];
            }

            if (transA == '!')
            {
                ConjugateInPlace(aCopy, n, n, lda);
                transA = 'N';
            }

            var b = new Complex[n];
            for (var i = 0; i < n; i++)
            {
                b[i] = x[i * incx];
            }

            var info = 0;
            fixed (Complex* pa = aCopy)
            fixed (Complex* pb = b)
            {
                lapack.ztrtrs(
                    uplo, transA, 'N',
                    n, 1,
                    (complex*)pa, lda,
                    (complex*)pb, n,
                    ref info);
            }

            Assert.That(info, Is.EqualTo(0));

            for (var i = 0; i < n; i++)
            {
                x[i * incx] = b[i];
            }
        }

        private static void ConjugateInPlace(Complex[] a, int rows, int cols, int lda)
        {
            for (var col = 0; col < cols; col++)
            {
                for (var row = 0; row < rows; row++)
                {
                    var index = (col * lda) + row;
                    var value = a[index];
                    a[index] = new Complex(value.Real, -value.Imaginary);
                }
            }
        }

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
        public unsafe void SolveTriangularComplex_UpperNoTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('U', 'N', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Upper, Transpose.NoTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }

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
        public unsafe void SolveTriangularComplex_LowerNoTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('L', 'N', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }

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
        public unsafe void SolveTriangularComplex_UpperTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('U', 'T', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Upper, Transpose.Trans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }

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
        public unsafe void SolveTriangularComplex_LowerTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('L', 'T', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.Trans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }

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
        public unsafe void SolveTriangularComplex_UpperConjNoTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('U', '!', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Upper, Transpose.ConjNoTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }

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
        public unsafe void SolveTriangularComplex_UpperConjTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('U', 'C', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Upper, Transpose.ConjTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }

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
        public unsafe void SolveTriangularComplex_LowerConjNoTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('L', '!', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.ConjNoTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }

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
        public unsafe void SolveTriangularComplex_LowerConjTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomComplex(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomComplex(57, n, incx);

            var expected = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = expected)
            {
                FakeZtrsv('L', 'C', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.ConjTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }
    }
}
