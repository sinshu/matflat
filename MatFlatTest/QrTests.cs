using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class QrTests
    {
        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 3, 4, 5)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 5, 4, 6)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 5, 7, 6)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 24, 20)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 37)]
        public unsafe void QrSingle_General(int m, int n, int lda, int ldq, int ldr)
        {
            var original = Matrix.RandomSingle(42, m, n, lda);

            var a = original.ToArray();
            var rdiag = new float[n];
            var q = Matrix.RandomSingle(0, m, n, ldq);
            var qCopy = q.ToArray();
            var r = Matrix.RandomSingle(0, n, n, ldr);
            var rCopy = r.ToArray();
            var reconstructed = new float[m * n];
            var identity = new float[n * n];
            fixed (float* pa = a)
            fixed (float* prdiag = rdiag)
            fixed (float* pq = q)
            fixed (float* pr = r)
            fixed (float* preconstructed = reconstructed)
            fixed (float* pidentity = identity)
            {
                Factorization.Qr(m, n, pa, lda, prdiag);
                Factorization.QrOrthogonalFactor(m, n, pa, lda, pq, ldq);
                Factorization.QrUpperTriangularFactor(m, n, pa, lda, pr, ldr, prdiag);
                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, n,
                    1.0F,
                    pq, ldq,
                    pr, ldr,
                    0.0F,
                    preconstructed, m);
                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans, OpenBlasSharp.Transpose.NoTrans,
                    n, n, m,
                    1.0F,
                    pq, ldq,
                    pq, ldq,
                    0.0F,
                    pidentity, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-4));
                }
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value, Is.EqualTo(1.0).Within(1.0E-4));
                    }
                    else
                    {
                        Assert.That(value, Is.EqualTo(0.0).Within(1.0E-5));
                    }
                }
            }

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= m)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
                }
            }

            for (var i = 0; i < q.Length; i++)
            {
                var row = i % ldq;
                var col = i / ldq;
                if (row >= m)
                {
                    Assert.That(q[i], Is.EqualTo(qCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < r.Length; i++)
            {
                var row = i % ldr;
                var col = i / ldr;
                if (row >= n)
                {
                    Assert.That(r[i], Is.EqualTo(rCopy[i]).Within(0));
                }
            }
        }

        [Test]
        public unsafe void QrSingle_Singular()
        {
            var m = 3;
            var n = 3;
            var lda = 3;
            var ldr = 3;
            var ldq = 3;
            var original = new float[] { 1, 2, 3, 2, 4, 6, 0, 0, 0 };

            var a = original.ToArray();
            var rdiag = new float[n];
            var q = Matrix.RandomSingle(0, m, n, ldq);
            var qCopy = q.ToArray();
            var r = Matrix.RandomSingle(0, n, n, ldr);
            var rCopy = r.ToArray();
            var reconstructed = new float[m * n];
            var identity = new float[n * n];
            fixed (float* pa = a)
            fixed (float* prdiag = rdiag)
            fixed (float* pq = q)
            fixed (float* pr = r)
            fixed (float* preconstructed = reconstructed)
            fixed (float* pidentity = identity)
            {
                Factorization.Qr(m, n, pa, lda, prdiag);
                Factorization.QrOrthogonalFactor(m, n, pa, lda, pq, ldq);
                Factorization.QrUpperTriangularFactor(m, n, pa, lda, pr, ldr, prdiag);
                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, n,
                    1.0F,
                    pq, ldq,
                    pr, ldr,
                    0.0F,
                    preconstructed, m);
                OpenBlasSharp.Blas.Sgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans, OpenBlasSharp.Transpose.NoTrans,
                    n, n, m,
                    1.0F,
                    pq, ldq,
                    pq, ldq,
                    0.0F,
                    pidentity, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-6));
                }
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value, Is.EqualTo(1.0).Within(1.0E-6));
                    }
                    else
                    {
                        Assert.That(value, Is.EqualTo(0.0).Within(1.0E-6));
                    }
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 3, 4, 5)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 5, 4, 6)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 5, 7, 6)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 24, 20)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 37)]
        public unsafe void QrDouble_General(int m, int n, int lda, int ldq, int ldr)
        {
            var original = Matrix.RandomDouble(42, m, n, lda);

            var a = original.ToArray();
            var rdiag = new double[n];
            var q = Matrix.RandomDouble(0, m, n, ldq);
            var qCopy = q.ToArray();
            var r = Matrix.RandomDouble(0, n, n, ldr);
            var rCopy = r.ToArray();
            var reconstructed = new double[m * n];
            var identity = new double[n * n];
            fixed (double* pa = a)
            fixed (double* prdiag = rdiag)
            fixed (double* pq = q)
            fixed (double* pr = r)
            fixed (double* preconstructed = reconstructed)
            fixed (double* pidentity = identity)
            {
                Factorization.Qr(m, n, pa, lda, prdiag);
                Factorization.QrOrthogonalFactor(m, n, pa, lda, pq, ldq);
                Factorization.QrUpperTriangularFactor(m, n, pa, lda, pr, ldr, prdiag);
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, n,
                    1.0,
                    pq, ldq,
                    pr, ldr,
                    0.0,
                    preconstructed, m);
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans, OpenBlasSharp.Transpose.NoTrans,
                    n, n, m,
                    1.0,
                    pq, ldq,
                    pq, ldq,
                    0.0,
                    pidentity, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
                }
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value, Is.EqualTo(1.0).Within(1.0E-12));
                    }
                    else
                    {
                        Assert.That(value, Is.EqualTo(0.0).Within(1.0E-12));
                    }
                }
            }

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= m)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
                }
            }

            for (var i = 0; i < q.Length; i++)
            {
                var row = i % ldq;
                var col = i / ldq;
                if (row >= m)
                {
                    Assert.That(q[i], Is.EqualTo(qCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < r.Length; i++)
            {
                var row = i % ldr;
                var col = i / ldr;
                if (row >= n)
                {
                    Assert.That(r[i], Is.EqualTo(rCopy[i]).Within(0));
                }
            }
        }

        [Test]
        public unsafe void QrDouble_Singular()
        {
            var m = 3;
            var n = 3;
            var lda = 3;
            var ldr = 3;
            var ldq = 3;
            var original = new double[] { 1, 2, 3, 2, 4, 6, 0, 0, 0 };

            var a = original.ToArray();
            var rdiag = new double[n];
            var q = Matrix.RandomDouble(0, m, n, ldq);
            var qCopy = q.ToArray();
            var r = Matrix.RandomDouble(0, n, n, ldr);
            var rCopy = r.ToArray();
            var reconstructed = new double[m * n];
            var identity = new double[n * n];
            fixed (double* pa = a)
            fixed (double* prdiag = rdiag)
            fixed (double* pq = q)
            fixed (double* pr = r)
            fixed (double* preconstructed = reconstructed)
            fixed (double* pidentity = identity)
            {
                Factorization.Qr(m, n, pa, lda, prdiag);
                Factorization.QrOrthogonalFactor(m, n, pa, lda, pq, ldq);
                Factorization.QrUpperTriangularFactor(m, n, pa, lda, pr, ldr, prdiag);
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, n,
                    1.0,
                    pq, ldq,
                    pr, ldr,
                    0.0,
                    preconstructed, m);
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.Trans, OpenBlasSharp.Transpose.NoTrans,
                    n, n, m,
                    1.0,
                    pq, ldq,
                    pq, ldq,
                    0.0,
                    pidentity, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-12));
                }
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value, Is.EqualTo(1.0).Within(1.0E-12));
                    }
                    else
                    {
                        Assert.That(value, Is.EqualTo(0.0).Within(1.0E-12));
                    }
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 3, 4, 5)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 5, 4, 6)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 5, 7, 6)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 24, 20)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 37)]
        public unsafe void QrComplex_General(int m, int n, int lda, int ldq, int ldr)
        {
            var original = Matrix.RandomComplex(42, m, n, lda);

            var a = original.ToArray();
            var rdiag = new double[n];
            var q = Matrix.RandomComplex(0, m, n, ldq);
            var qCopy = q.ToArray();
            var r = Matrix.RandomComplex(0, n, n, ldr);
            var rCopy = r.ToArray();
            var reconstructed = new Complex[m * n];
            var identity = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (double* prdiag = rdiag)
            fixed (Complex* pq = q)
            fixed (Complex* pr = r)
            fixed (Complex* preconstructed = reconstructed)
            fixed (Complex* pidentity = identity)
            {
                var one = Complex.One;
                var zero = Complex.Zero;
                Factorization.Qr(m, n, pa, lda, prdiag);
                Factorization.QrOrthogonalFactor(m, n, pa, lda, pq, ldq);
                Factorization.QrUpperTriangularFactor(m, n, pa, lda, pr, ldr, prdiag);
                OpenBlasSharp.Blas.Zgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, n,
                    &one,
                    pq, ldq,
                    pr, ldr,
                    &zero,
                    preconstructed, m);
                OpenBlasSharp.Blas.Zgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.ConjTrans, OpenBlasSharp.Transpose.NoTrans,
                    n, n, m,
                    &one,
                    pq, ldq,
                    pq, ldq,
                    &zero,
                    pidentity, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual.Real, Is.EqualTo(expected.Real).Within(1.0E-12));
                    Assert.That(actual.Imaginary, Is.EqualTo(expected.Imaginary).Within(1.0E-12));
                }
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value.Real, Is.EqualTo(1.0).Within(1.0E-12));
                    }
                    else
                    {
                        Assert.That(value.Real, Is.EqualTo(0.0).Within(1.0E-12));
                    }
                    Assert.That(value.Imaginary, Is.EqualTo(0.0).Within(1.0E-12));
                }
            }

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= m)
                {
                    Assert.That(a[i].Real, Is.EqualTo(original[i].Real).Within(0));
                    Assert.That(a[i].Imaginary, Is.EqualTo(original[i].Imaginary).Within(0));
                }
            }

            for (var i = 0; i < q.Length; i++)
            {
                var row = i % ldq;
                var col = i / ldq;
                if (row >= m)
                {
                    Assert.That(q[i].Real, Is.EqualTo(qCopy[i].Real).Within(0));
                    Assert.That(q[i].Imaginary, Is.EqualTo(qCopy[i].Imaginary).Within(0));
                }
            }

            for (var i = 0; i < r.Length; i++)
            {
                var row = i % ldr;
                var col = i / ldr;
                if (row >= n)
                {
                    Assert.That(r[i].Real, Is.EqualTo(rCopy[i].Real).Within(0));
                    Assert.That(r[i].Imaginary, Is.EqualTo(rCopy[i].Imaginary).Within(0));
                }
            }
        }

        [Test]
        public unsafe void QrComplex_Singular()
        {
            var j = Complex.ImaginaryOne;
            var m = 3;
            var n = 3;
            var lda = 3;
            var ldr = 3;
            var ldq = 3;
            var original = new Complex[] { 1, 2, 3, 2 * j, 4 * j, 6 * j, 0, 0, 0 };

            var a = original.ToArray();
            var rdiag = new double[n];
            var q = Matrix.RandomComplex(0, m, n, ldq);
            var qCopy = q.ToArray();
            var r = Matrix.RandomComplex(0, n, n, ldr);
            var rCopy = r.ToArray();
            var reconstructed = new Complex[m * n];
            var identity = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (double* prdiag = rdiag)
            fixed (Complex* pq = q)
            fixed (Complex* pr = r)
            fixed (Complex* preconstructed = reconstructed)
            fixed (Complex* pidentity = identity)
            {
                var one = Complex.One;
                var zero = Complex.Zero;
                Factorization.Qr(m, n, pa, lda, prdiag);
                Factorization.QrOrthogonalFactor(m, n, pa, lda, pq, ldq);
                Factorization.QrUpperTriangularFactor(m, n, pa, lda, pr, ldr, prdiag);
                OpenBlasSharp.Blas.Zgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans, OpenBlasSharp.Transpose.NoTrans,
                    m, n, n,
                    &one,
                    pq, ldq,
                    pr, ldr,
                    &zero,
                    preconstructed, m);
                OpenBlasSharp.Blas.Zgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.ConjTrans, OpenBlasSharp.Transpose.NoTrans,
                    n, n, m,
                    &one,
                    pq, ldq,
                    pq, ldq,
                    &zero,
                    pidentity, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual.Real, Is.EqualTo(expected.Real).Within(1.0E-12));
                    Assert.That(actual.Imaginary, Is.EqualTo(expected.Imaginary).Within(1.0E-12));
                }
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity[col * n + row];
                    if (row == col)
                    {
                        Assert.That(value.Real, Is.EqualTo(1.0).Within(1.0E-12));
                    }
                    else
                    {
                        Assert.That(value.Real, Is.EqualTo(0.0).Within(1.0E-12));
                    }
                    Assert.That(value.Imaginary, Is.EqualTo(0.0).Within(1.0E-12));
                }
            }
        }
    }
}
