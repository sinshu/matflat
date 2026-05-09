using System;
using System.Linq;
using ILNumerics.Core.Native;
using ILNumerics.F2NET;
using MatFlat;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_SolveTriangularSingle
    {
        private static readonly ILapack lapack = new ManagedLAPACK();

        private static unsafe void FakeStrsv(char uplo, char transA, int n, float* a, int lda, float* x, int incx)
        {
            var b = new float[n];
            for (var i = 0; i < n; i++)
            {
                b[i] = x[i * incx];
            }

            var info = 0;
            fixed (float* pb = b)
            {
                lapack.strtrs(
                    uplo, transA, 'N',
                    n, 1,
                    a, lda,
                    pb, n,
                    ref info);
            }

            Assert.That(info, Is.EqualTo(0));

            for (var i = 0; i < n; i++)
            {
                x[i * incx] = b[i];
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
        public unsafe void SolveTriangularSingle_UpperNoTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomSingle(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomSingle(57, n, incx);

            var expected = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = expected)
            {
                FakeStrsv('U', 'N', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = actual)
            {
                Blas.SolveTriangular(Uplo.Upper, Transpose.NoTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-3));
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
        public unsafe void SolveTriangularSingle_LowerNoTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomSingle(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomSingle(57, n, incx);

            var expected = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = expected)
            {
                FakeStrsv('L', 'N', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = actual)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, pa, lda, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-3));
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
        public unsafe void SolveTriangularSingle_UpperTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomSingle(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomSingle(57, n, incx);

            var expected = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = expected)
            {
                FakeStrsv('U', 'T', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = actual)
            {
                Blas.SolveTriangular(Uplo.Upper, Transpose.Trans, n, pa, lda, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-3));
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
        public unsafe void SolveTriangularSingle_LowerTrans(int n, int lda, int incx)
        {
            var a = Matrix.RandomSingle(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < row; col++)
                {
                    Matrix.Set(n, n, a, lda, row, col, 0);
                }
            }

            var input = Vector.RandomSingle(57, n, incx);

            var expected = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = expected)
            {
                FakeStrsv('L', 'T', n, pa, lda, px, incx);
            }

            var actual = input.ToArray();
            fixed (float* pa = a)
            fixed (float* px = actual)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.Trans, n, pa, lda, px, incx);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-3));
        }
    }
}
