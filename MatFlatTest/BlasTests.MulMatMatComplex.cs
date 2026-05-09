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
    public class BlasTests_MulMatMatComplex
    {
        private static readonly ILapack lapack = new ManagedLAPACK();

        private static unsafe void FakeZgemm(char TransA, char TransB, int M, int N, int K, Complex alpha, Complex* A, int lda, Complex* B, int ldb, Complex beta, Complex* C, int ldc)
        {
            // ManagedLAPACK does not support OpenBLAS' ConjNoTrans mode, so this
            // test helper treats '!' as "conjugate without transpose".
            if (TransA == '!')
            {
                ConjugateInPlace(A, M, K, lda);
                TransA = 'N';
            }

            if (TransB == '!')
            {
                ConjugateInPlace(B, K, N, ldb);
                TransB = 'N';
            }

            lapack.zgemm(
                TransA, TransB,
                M, N, K,
                ToLapackComplex(alpha),
                (complex*)A, lda,
                (complex*)B, ldb,
                ToLapackComplex(beta),
                (complex*)C, ldc);
        }

        private static unsafe void ConjugateInPlace(Complex* a, int rows, int cols, int lda)
        {
            for (var col = 0; col < cols; col++)
            {
                for (var row = 0; row < rows; row++)
                {
                    var value = a[(col * lda) + row];
                    a[(col * lda) + row] = new Complex(value.Real, -value.Imaginary);
                }
            }
        }

        private static complex ToLapackComplex(Complex value)
        {
            return new complex(value.Real, value.Imaginary);
        }

        [TestCase(1, 1, 1, 1, 1, 1)]
        [TestCase(1, 1, 1, 2, 3, 4)]
        [TestCase(2, 2, 2, 2, 2, 2)]
        [TestCase(2, 2, 2, 3, 4, 5)]
        [TestCase(3, 3, 3, 3, 3, 3)]
        [TestCase(3, 3, 3, 4, 5, 6)]
        [TestCase(2, 3, 4, 2, 4, 2)]
        [TestCase(2, 3, 4, 3, 5, 3)]
        [TestCase(1, 5, 2, 1, 2, 1)]
        [TestCase(1, 5, 2, 3, 4, 3)]
        [TestCase(5, 3, 2, 5, 2, 5)]
        [TestCase(5, 3, 2, 7, 4, 7)]
        public unsafe void NN(int m, int n, int k, int lda, int ldb, int ldc)
        {
            for (var cond = 0; cond < 4; cond++)
            {
                var (lapackTrans1, lapackTrans2, mfTrans1, mfTrans2) = GetCondition(false, false, cond);

                var a = Matrix.RandomComplex(42, m, k, lda);
                var b = Matrix.RandomComplex(57, k, n, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pb = b.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    FakeZgemm(
                        lapackTrans1, lapackTrans2,
                        m, n, k,
                        one,
                        pa, lda,
                        pb, ldb,
                        zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
                }

                Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
                Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
            }
        }

        [TestCase(1, 1, 1, 1, 1, 1)]
        [TestCase(2, 2, 2, 2, 2, 2)]
        [TestCase(3, 3, 3, 3, 3, 3)]
        [TestCase(2, 3, 4, 4, 4, 2)]
        [TestCase(2, 3, 4, 5, 6, 3)]
        [TestCase(5, 4, 3, 3, 3, 5)]
        [TestCase(5, 4, 3, 4, 5, 6)]
        [TestCase(3, 7, 5, 5, 5, 3)]
        [TestCase(3, 7, 5, 8, 9, 7)]
        public unsafe void TN(int m, int n, int k, int lda, int ldb, int ldc)
        {
            for (var cond = 0; cond < 4; cond++)
            {
                var (lapackTrans1, lapackTrans2, mfTrans1, mfTrans2) = GetCondition(true, false, cond);

                var a = Matrix.RandomComplex(42, k, m, lda);
                var b = Matrix.RandomComplex(57, k, n, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pb = b.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    FakeZgemm(
                        lapackTrans1, lapackTrans2,
                        m, n, k,
                        one,
                        pa, lda,
                        pb, ldb,
                        zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
                }

                Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
                Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
            }
        }

        [TestCase(1, 1, 1, 1, 1, 1)]
        [TestCase(2, 2, 2, 2, 2, 2)]
        [TestCase(3, 3, 3, 3, 3, 3)]
        [TestCase(2, 3, 4, 2, 3, 2)]
        [TestCase(2, 3, 4, 5, 6, 3)]
        [TestCase(5, 4, 3, 5, 4, 5)]
        [TestCase(5, 4, 3, 6, 6, 7)]
        [TestCase(3, 7, 5, 3, 7, 3)]
        [TestCase(3, 7, 5, 5, 9, 6)]
        public unsafe void NT(int m, int n, int k, int lda, int ldb, int ldc)
        {
            for (var cond = 0; cond < 4; cond++)
            {
                var (lapackTrans1, lapackTrans2, mfTrans1, mfTrans2) = GetCondition(false, true, cond);

                var a = Matrix.RandomComplex(42, m, k, lda);
                var b = Matrix.RandomComplex(57, n, k, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pb = b.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    FakeZgemm(
                        lapackTrans1, lapackTrans2,
                        m, n, k,
                        one,
                        pa, lda,
                        pb, ldb,
                        zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
                }

                Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
                Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
            }
        }

        [TestCase(1, 1, 1, 1, 1, 1)]
        [TestCase(2, 2, 2, 2, 2, 2)]
        [TestCase(3, 3, 3, 3, 3, 3)]
        [TestCase(2, 3, 4, 4, 3, 2)]
        [TestCase(2, 3, 4, 5, 5, 5)]
        [TestCase(5, 4, 3, 3, 4, 5)]
        [TestCase(5, 4, 3, 7, 5, 6)]
        [TestCase(3, 7, 5, 5, 7, 3)]
        [TestCase(3, 7, 5, 6, 8, 4)]
        public unsafe void TT(int m, int n, int k, int lda, int ldb, int ldc)
        {
            for (var cond = 0; cond < 4; cond++)
            {
                var (lapackTrans1, lapackTrans2, mfTrans1, mfTrans2) = GetCondition(true, true, cond);

                var a = Matrix.RandomComplex(42, k, m, lda);
                var b = Matrix.RandomComplex(57, n, k, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pb = b.ToArray()) // FakeZgemm might modify the input matrices, so we pass copies of them.
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    FakeZgemm(
                        lapackTrans1, lapackTrans2,
                        m, n, k,
                        one,
                        pa, lda,
                        pb, ldb,
                        zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
                }

                Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
                Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
            }
        }

        private static (char, char, Transpose, Transpose) GetCondition(bool transa, bool transb, int cond)
        {
            switch (cond)
            {
                case 0:
                    return (
                        transa ? 'T' : 'N',
                        transb ? 'T' : 'N',
                        transa ? Transpose.Trans : Transpose.NoTrans,
                        transb ? Transpose.Trans : Transpose.NoTrans);
                case 1:
                    return (
                        AddConj(transa ? 'T' : 'N'),
                        transb ? 'T' : 'N',
                        AddConj(transa ? Transpose.Trans : Transpose.NoTrans),
                        transb ? Transpose.Trans : Transpose.NoTrans);
                case 2:
                    return (
                        transa ? 'T' : 'N',
                        AddConj(transb ? 'T' : 'N'),
                        transa ? Transpose.Trans : Transpose.NoTrans,
                        AddConj(transb ? Transpose.Trans : Transpose.NoTrans));
                case 3:
                    return (
                        AddConj(transa ? 'T' : 'N'),
                        AddConj(transb ? 'T' : 'N'),
                        AddConj(transa ? Transpose.Trans : Transpose.NoTrans),
                        AddConj(transb ? Transpose.Trans : Transpose.NoTrans));
                default:
                    throw new Exception();
            }
        }

        private static char AddConj(char trans)
        {
            if (trans == 'N')
            {
                return '!';
            }
            else if (trans == 'T')
            {
                return 'C';
            }
            else
            {
                throw new Exception();
            }
        }

        private static Transpose AddConj(Transpose trans)
        {
            if (trans == Transpose.NoTrans)
            {
                return Transpose.ConjNoTrans;
            }
            else if (trans == Transpose.Trans)
            {
                return Transpose.ConjTrans;
            }
            else
            {
                throw new Exception();
            }
        }
    }
}
