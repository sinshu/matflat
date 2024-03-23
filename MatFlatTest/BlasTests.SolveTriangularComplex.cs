using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_SolveTriangularComplex
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
                OpenBlasSharp.Blas.Ztrsv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Uplo.Upper,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Diag.NonUnit,
                    n,
                    pa, lda,
                    px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                MatFlat.Blas.SolveTriangular(MatFlat.Uplo.Upper, MatFlat.Transpose.NoTrans, n, pa, lda, px, incx);
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
                OpenBlasSharp.Blas.Ztrsv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Uplo.Lower,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Diag.NonUnit,
                    n,
                    pa, lda,
                    px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                MatFlat.Blas.SolveTriangular(MatFlat.Uplo.Lower, MatFlat.Transpose.NoTrans, n, pa, lda, px, incx);
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
                OpenBlasSharp.Blas.Ztrsv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Uplo.Upper,
                    OpenBlasSharp.Transpose.Trans,
                    OpenBlasSharp.Diag.NonUnit,
                    n,
                    pa, lda,
                    px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                MatFlat.Blas.SolveTriangular(MatFlat.Uplo.Upper, MatFlat.Transpose.Trans, n, pa, lda, px, incx);
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
                OpenBlasSharp.Blas.Ztrsv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Uplo.Lower,
                    OpenBlasSharp.Transpose.Trans,
                    OpenBlasSharp.Diag.NonUnit,
                    n,
                    pa, lda,
                    px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                MatFlat.Blas.SolveTriangular(MatFlat.Uplo.Lower, MatFlat.Transpose.Trans, n, pa, lda, px, incx);
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
                OpenBlasSharp.Blas.Ztrsv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Uplo.Upper,
                    OpenBlasSharp.Transpose.ConjNoTrans,
                    OpenBlasSharp.Diag.NonUnit,
                    n,
                    pa, lda,
                    px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                MatFlat.Blas.SolveTriangular(MatFlat.Uplo.Upper, MatFlat.Transpose.ConjNoTrans, n, pa, lda, px, incx);
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
                OpenBlasSharp.Blas.Ztrsv(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Uplo.Upper,
                    OpenBlasSharp.Transpose.ConjTrans,
                    OpenBlasSharp.Diag.NonUnit,
                    n,
                    pa, lda,
                    px, incx);
            }

            var actual = input.ToArray();
            fixed (Complex* pa = a)
            fixed (Complex* px = actual)
            {
                MatFlat.Blas.SolveTriangular(MatFlat.Uplo.Upper, MatFlat.Transpose.ConjTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-11));
            Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-11));
        }
    }
}
