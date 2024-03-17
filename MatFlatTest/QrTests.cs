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
        [TestCase(1, 1, 1)]
        [TestCase(1, 1, 3)]
        [TestCase(2, 2, 2)]
        [TestCase(2, 2, 3)]
        [TestCase(3, 3, 3)]
        [TestCase(3, 3, 5)]
        [TestCase(3, 1, 3)]
        [TestCase(3, 1, 5)]
        [TestCase(4, 3, 4)]
        [TestCase(4, 3, 5)]
        [TestCase(16, 8, 16)]
        [TestCase(16, 8, 32)]
        [TestCase(23, 11, 23)]
        [TestCase(23, 11, 31)]
        public unsafe void QrDouble_General(int m, int n, int lda)
        {
            var a = Matrix.RandomDouble(42, m, n, lda);

            var expectedA = a.ToArray();
            var expectedTau = new double[n];
            fixed (double* pa = expectedA)
            fixed (double* ptau = expectedTau)
            {
                Lapack.Dgeqrf(MatrixLayout.ColMajor, m, n, pa, lda, ptau);
            }

            var actualA = a.ToArray();
            var actualRdiag = new double[n];
            fixed (double* pa = actualA)
            fixed (double* prdiag = actualRdiag)
            {
                Factorization.QrDouble(m, n, pa, lda, prdiag);
            }

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (col > row)
                {
                    Assert.That(actualA[i], Is.EqualTo(expectedA[i]).Within(1.0E-12));
                }
                else if (row == col)
                {
                    Assert.That(Math.Abs(actualRdiag[col]), Is.EqualTo(Math.Abs(expectedA[i])).Within(1.0E-12));
                }
                else if (row >= m)
                {
                    Assert.That(actualA[i], Is.EqualTo(expectedA[i]).Within(0));
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
        public unsafe void QrDouble_General2(int m, int n, int lda, int ldq, int ldr)
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
                Factorization.QrDouble(m, n, pa, lda, prdiag);
                Factorization.QrOrthogonalFactorDouble(m, n, pa, lda, pq, ldq);
                Factorization.QrUpperTriangularFactorDouble(m, n, pa, lda, pr, ldr, prdiag);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0,
                    pq, ldq,
                    pr, ldr,
                    0.0,
                    preconstructed, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
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
    }
}
