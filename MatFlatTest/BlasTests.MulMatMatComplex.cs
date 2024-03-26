using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_MulMatMatComplex
    {
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
                var (obTrans1, obTrans2, mfTrans1, mfTrans2) = GetCondition(false, false, cond);

                var a = Matrix.RandomComplex(42, m, k, lda);
                var b = Matrix.RandomComplex(57, k, n, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    OpenBlasSharp.Blas.Zgemm(
                        OpenBlasSharp.Order.ColMajor,
                        obTrans1, obTrans2,
                        m, n, k,
                        &one,
                        pa, lda,
                        pb, ldb,
                        &zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    MatFlat.Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
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
                var (obTrans1, obTrans2, mfTrans1, mfTrans2) = GetCondition(true, false, cond);

                var a = Matrix.RandomComplex(42, k, m, lda);
                var b = Matrix.RandomComplex(57, k, n, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    OpenBlasSharp.Blas.Zgemm(
                        OpenBlasSharp.Order.ColMajor,
                        obTrans1, obTrans2,
                        m, n, k,
                        &one,
                        pa, lda,
                        pb, ldb,
                        &zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    MatFlat.Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
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
                var (obTrans1, obTrans2, mfTrans1, mfTrans2) = GetCondition(false, true, cond);

                var a = Matrix.RandomComplex(42, m, k, lda);
                var b = Matrix.RandomComplex(57, n, k, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    OpenBlasSharp.Blas.Zgemm(
                        OpenBlasSharp.Order.ColMajor,
                        obTrans1, obTrans2,
                        m, n, k,
                        &one,
                        pa, lda,
                        pb, ldb,
                        &zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    MatFlat.Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
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
                var (obTrans1, obTrans2, mfTrans1, mfTrans2) = GetCondition(true, true, cond);

                var a = Matrix.RandomComplex(42, k, m, lda);
                var b = Matrix.RandomComplex(57, n, k, ldb);
                var c = Matrix.RandomComplex(0, m, n, ldc);

                var expected = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = expected)
                {
                    var one = Complex.One;
                    var zero = Complex.Zero;

                    OpenBlasSharp.Blas.Zgemm(
                        OpenBlasSharp.Order.ColMajor,
                        obTrans1, obTrans2,
                        m, n, k,
                        &one,
                        pa, lda,
                        pb, ldb,
                        &zero,
                        pc, ldc);
                }

                var actual = c.ToArray();
                fixed (Complex* pa = a)
                fixed (Complex* pb = b)
                fixed (Complex* pc = actual)
                {
                    MatFlat.Blas.MulMatMat(mfTrans1, mfTrans2, m, n, k, pa, lda, pb, ldb, pc, ldc);
                }

                Assert.That(actual.Select(x => x.Real), Is.EqualTo(expected.Select(x => x.Real)).Within(1.0E-12));
                Assert.That(actual.Select(x => x.Imaginary), Is.EqualTo(expected.Select(x => x.Imaginary)).Within(1.0E-12));
            }
        }

        private static (OpenBlasSharp.Transpose, OpenBlasSharp.Transpose, MatFlat.Transpose, MatFlat.Transpose) GetCondition(bool transa, bool transb, int cond)
        {
            switch (cond)
            {
                case 0:
                    return (
                        transa ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans,
                        transb ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans,
                        transa ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans,
                        transb ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans);
                case 1:
                    return (
                        AddConj(transa ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans),
                        transb ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans,
                        AddConj(transa ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans),
                        transb ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans);
                case 2:
                    return (
                        transa ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans,
                        AddConj(transb ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans),
                        transa ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans,
                        AddConj(transb ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans));
                case 3:
                    return (
                        AddConj(transa ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans),
                        AddConj(transb ? OpenBlasSharp.Transpose.Trans : OpenBlasSharp.Transpose.NoTrans),
                        AddConj(transa ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans),
                        AddConj(transb ? MatFlat.Transpose.Trans : MatFlat.Transpose.NoTrans));
                default:
                    throw new Exception();
            }
        }

        private static OpenBlasSharp.Transpose AddConj(OpenBlasSharp.Transpose trans)
        {
            if (trans == OpenBlasSharp.Transpose.NoTrans)
            {
                return OpenBlasSharp.Transpose.ConjNoTrans;
            }
            else if (trans == OpenBlasSharp.Transpose.Trans)
            {
                return OpenBlasSharp.Transpose.ConjTrans;
            }
            else
            {
                throw new Exception();
            }
        }

        private static MatFlat.Transpose AddConj(MatFlat.Transpose trans)
        {
            if (trans == MatFlat.Transpose.NoTrans)
            {
                return MatFlat.Transpose.ConjNoTrans;
            }
            else if (trans == MatFlat.Transpose.Trans)
            {
                return MatFlat.Transpose.ConjTrans;
            }
            else
            {
                throw new Exception();
            }
        }
    }
}
