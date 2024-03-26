using System;
using System.Linq;
using NUnit.Framework;

namespace MatFlatTest
{
    public class BlasTests_MulMatMatSingle
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
            var a = Matrix.RandomSingle(42, m, k, lda);
            var b = Matrix.RandomSingle(57, k, n, ldb);
            var c = Matrix.RandomSingle(0, m, n, ldc);

            var expected = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = expected)
            {
                OpenBlasSharp.Blas.Sgemm(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, k,
                    1.0F,
                    pa, lda,
                    pb, ldb,
                    0.0F,
                    pc, ldc);
            }

            var actual = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = actual)
            {
                MatFlat.Blas.MulMatMat(MatFlat.Transpose.NoTrans, MatFlat.Transpose.NoTrans, m, n, k, pa, lda, pb, ldb, pc, ldc);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
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
            var a = Matrix.RandomSingle(42, k, m, lda);
            var b = Matrix.RandomSingle(57, k, n, ldb);
            var c = Matrix.RandomSingle(0, m, n, ldc);

            var expected = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = expected)
            {
                OpenBlasSharp.Blas.Sgemm(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, k,
                    1.0F,
                    pa, lda,
                    pb, ldb,
                    0.0F,
                    pc, ldc);
            }

            var actual = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = actual)
            {
                MatFlat.Blas.MulMatMat(MatFlat.Transpose.Trans, MatFlat.Transpose.NoTrans, m, n, k, pa, lda, pb, ldb, pc, ldc);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
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
            var a = Matrix.RandomSingle(42, m, k, lda);
            var b = Matrix.RandomSingle(57, n, k, ldb);
            var c = Matrix.RandomSingle(0, m, n, ldc);

            var expected = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = expected)
            {
                OpenBlasSharp.Blas.Sgemm(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.Trans,
                    m, n, k,
                    1.0F,
                    pa, lda,
                    pb, ldb,
                    0.0F,
                    pc, ldc);
            }

            var actual = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = actual)
            {
                MatFlat.Blas.MulMatMat(MatFlat.Transpose.NoTrans, MatFlat.Transpose.Trans, m, n, k, pa, lda, pb, ldb, pc, ldc);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
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
            var a = Matrix.RandomSingle(42, k, m, lda);
            var b = Matrix.RandomSingle(57, n, k, ldb);
            var c = Matrix.RandomSingle(0, m, n, ldc);

            var expected = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = expected)
            {
                OpenBlasSharp.Blas.Sgemm(
                    OpenBlasSharp.Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans, OpenBlasSharp.Transpose.Trans,
                    m, n, k,
                    1.0F,
                    pa, lda,
                    pb, ldb,
                    0.0F,
                    pc, ldc);
            }

            var actual = c.ToArray();
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = actual)
            {
                MatFlat.Blas.MulMatMat(MatFlat.Transpose.Trans, MatFlat.Transpose.Trans, m, n, k, pa, lda, pb, ldb, pc, ldc);
            }

            Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
        }
    }
}
