using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class SvdTests
    {
        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdSingle_General(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomSingle(42, m, n, lda);

            var a = original.ToArray();
            var s = new float[Math.Min(m, n)];
            var u = Matrix.RandomSingle(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomSingle(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new float[m * n];
            var us = new float[m * n];
            var reconstructed = new float[m * n];
            var identity1 = new float[m * m];
            var identity2 = new float[n * n];
            fixed (float* pa = a)
            fixed (float* ps = s)
            fixed (float* pu = u)
            fixed (float* pvt = vt)
            fixed (float* psmat = smat)
            fixed (float* pus = us)
            fixed (float* preconstructed = reconstructed)
            fixed (float* pidentity1 = identity1)
            fixed (float* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0F,
                    pu, ldu,
                    psmat, m,
                    0.0F,
                    pus, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0F,
                    pus, m,
                    pvt, ldvt,
                    0.0F,
                    preconstructed, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0F,
                    pu, ldu,
                    pu, ldu,
                    0.0F,
                    pidentity1, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0F,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0F,
                    pidentity2, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-5));
                }
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= m)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
                }
            }

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdSingle_VectorOption(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomSingle(42, m, n, lda);

            var a = original.ToArray();
            var s = new float[Math.Min(m, n)];
            var u = Matrix.RandomSingle(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomSingle(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new float[m * n];
            var us = new float[m * n];
            var reconstructed = new float[m * n];
            var identity1 = new float[m * m];
            var identity2 = new float[n * n];
            fixed (float* pa = a)
            fixed (float* ps = s)
            fixed (float* pu = u)
            fixed (float* pvt = vt)
            fixed (float* psmat = smat)
            fixed (float* pus = us)
            fixed (float* preconstructed = reconstructed)
            fixed (float* pidentity1 = identity1)
            fixed (float* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, null, 0, null, 0);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                original.CopyTo(a, 0);
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, null, 0);

                original.CopyTo(a, 0);
                Factorization.Svd(m, n, pa, lda, ps, null, 0, pvt, ldvt);

                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0F,
                    pu, ldu,
                    psmat, m,
                    0.0F,
                    pus, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0F,
                    pus, m,
                    pvt, ldvt,
                    0.0F,
                    preconstructed, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0F,
                    pu, ldu,
                    pu, ldu,
                    0.0F,
                    pidentity1, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0F,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0F,
                    pidentity2, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-5));
                }
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= m)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
                }
            }

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdSingle_Singular(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomSingle(42, m, n, lda);
            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (row >= m / 2 || col >= n / 2)
                    {
                        Matrix.Set(m, n, original, lda, row, col, 0);
                    }
                }
            }

            var a = original.ToArray();
            var s = new float[Math.Min(m, n)];
            var u = Matrix.RandomSingle(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomSingle(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new float[m * n];
            var us = new float[m * n];
            var reconstructed = new float[m * n];
            var identity1 = new float[m * m];
            var identity2 = new float[n * n];
            fixed (float* pa = a)
            fixed (float* ps = s)
            fixed (float* pu = u)
            fixed (float* pvt = vt)
            fixed (float* psmat = smat)
            fixed (float* pus = us)
            fixed (float* preconstructed = reconstructed)
            fixed (float* pidentity1 = identity1)
            fixed (float* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0F,
                    pu, ldu,
                    psmat, m,
                    0.0F,
                    pus, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0F,
                    pus, m,
                    pvt, ldvt,
                    0.0F,
                    preconstructed, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0F,
                    pu, ldu,
                    pu, ldu,
                    0.0F,
                    pidentity1, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0F,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0F,
                    pidentity2, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-5));
                }
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= m)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
                }
            }

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [Test]
        public unsafe void SvdSingle_Zero()
        {
            var n = 3;
            var m = 3;
            var lda = 3;
            var ldu = 3;
            var ldvt = 3;
            var original = new float[m * n];

            var a = original.ToArray();
            var s = new float[Math.Min(m, n)];
            var u = Matrix.RandomSingle(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomSingle(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new float[m * n];
            var us = new float[m * n];
            var reconstructed = new float[m * n];
            var identity1 = new float[m * m];
            var identity2 = new float[n * n];
            fixed (float* pa = a)
            fixed (float* ps = s)
            fixed (float* pu = u)
            fixed (float* pvt = vt)
            fixed (float* psmat = smat)
            fixed (float* pus = us)
            fixed (float* preconstructed = reconstructed)
            fixed (float* pidentity1 = identity1)
            fixed (float* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0F,
                    pu, ldu,
                    psmat, m,
                    0.0F,
                    pus, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0F,
                    pus, m,
                    pvt, ldvt,
                    0.0F,
                    preconstructed, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0F,
                    pu, ldu,
                    pu, ldu,
                    0.0F,
                    pidentity1, m);
                Blas.Sgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0F,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0F,
                    pidentity2, n);
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var actual = Matrix.Get(m, n, reconstructed, m, row, col);
                    var expected = Matrix.Get(m, n, original, lda, row, col);
                    Assert.That(actual, Is.EqualTo(expected).Within(1.0E-5));
                }
            }

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < a.Length; i++)
            {
                var row = i % lda;
                var col = i / lda;
                if (row >= m)
                {
                    Assert.That(a[i], Is.EqualTo(original[i]).Within(0));
                }
            }

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdDouble_General(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomDouble(42, m, n, lda);

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomDouble(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomDouble(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new double[m * n];
            var us = new double[m * n];
            var reconstructed = new double[m * n];
            var identity1 = new double[m * m];
            var identity2 = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = s)
            fixed (double* pu = u)
            fixed (double* pvt = vt)
            fixed (double* psmat = smat)
            fixed (double* pus = us)
            fixed (double* preconstructed = reconstructed)
            fixed (double* pidentity1 = identity1)
            fixed (double* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0,
                    pu, ldu,
                    psmat, m,
                    0.0,
                    pus, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0,
                    pus, m,
                    pvt, ldvt,
                    0.0,
                    preconstructed, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0,
                    pu, ldu,
                    pu, ldu,
                    0.0,
                    pidentity1, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdDouble_VectorOption(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomDouble(42, m, n, lda);

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomDouble(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomDouble(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new double[m * n];
            var us = new double[m * n];
            var reconstructed = new double[m * n];
            var identity1 = new double[m * m];
            var identity2 = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = s)
            fixed (double* pu = u)
            fixed (double* pvt = vt)
            fixed (double* psmat = smat)
            fixed (double* pus = us)
            fixed (double* preconstructed = reconstructed)
            fixed (double* pidentity1 = identity1)
            fixed (double* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, null, 0, null, 0);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                original.CopyTo(a, 0);
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, null, 0);

                original.CopyTo(a, 0);
                Factorization.Svd(m, n, pa, lda, ps, null, 0, pvt, ldvt);

                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0,
                    pu, ldu,
                    psmat, m,
                    0.0,
                    pus, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0,
                    pus, m,
                    pvt, ldvt,
                    0.0,
                    preconstructed, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0,
                    pu, ldu,
                    pu, ldu,
                    0.0,
                    pidentity1, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdDouble_Singular(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomDouble(42, m, n, lda);
            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (row >= m / 2 || col >= n / 2)
                    {
                        Matrix.Set(m, n, original, lda, row, col, 0);
                    }
                }
            }

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomDouble(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomDouble(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new double[m * n];
            var us = new double[m * n];
            var reconstructed = new double[m * n];
            var identity1 = new double[m * m];
            var identity2 = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = s)
            fixed (double* pu = u)
            fixed (double* pvt = vt)
            fixed (double* psmat = smat)
            fixed (double* pus = us)
            fixed (double* preconstructed = reconstructed)
            fixed (double* pidentity1 = identity1)
            fixed (double* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0,
                    pu, ldu,
                    psmat, m,
                    0.0,
                    pus, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0,
                    pus, m,
                    pvt, ldvt,
                    0.0,
                    preconstructed, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0,
                    pu, ldu,
                    pu, ldu,
                    0.0,
                    pidentity1, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [Test]
        public unsafe void SvdDouble_Zero()
        {
            var n = 3;
            var m = 3;
            var lda = 3;
            var ldu = 3;
            var ldvt = 3;
            var original = new double[m * n];

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomDouble(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomDouble(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new double[m * n];
            var us = new double[m * n];
            var reconstructed = new double[m * n];
            var identity1 = new double[m * m];
            var identity2 = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = s)
            fixed (double* pu = u)
            fixed (double* pvt = vt)
            fixed (double* psmat = smat)
            fixed (double* pus = us)
            fixed (double* preconstructed = reconstructed)
            fixed (double* pidentity1 = identity1)
            fixed (double* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    1.0,
                    pu, ldu,
                    psmat, m,
                    0.0,
                    pus, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    1.0,
                    pus, m,
                    pvt, ldvt,
                    0.0,
                    preconstructed, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    m, m, m,
                    1.0,
                    pu, ldu,
                    pu, ldu,
                    0.0,
                    pidentity1, m);
                Blas.Dgemm(
                    Order.ColMajor,
                    Transpose.Trans, Transpose.NoTrans,
                    n, n, n,
                    1.0,
                    pvt, ldvt,
                    pvt, ldvt,
                    0.0,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i], Is.EqualTo(uCopy[i]).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i], Is.EqualTo(vtCopy[i]).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdComplex_General(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomComplex(42, m, n, lda);

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomComplex(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomComplex(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new Complex[m * n];
            var us = new Complex[m * n];
            var reconstructed = new Complex[m * n];
            var identity1 = new Complex[m * m];
            var identity2 = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (double* ps = s)
            fixed (Complex* pu = u)
            fixed (Complex* pvt = vt)
            fixed (Complex* psmat = smat)
            fixed (Complex* pus = us)
            fixed (Complex* preconstructed = reconstructed)
            fixed (Complex* pidentity1 = identity1)
            fixed (Complex* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                var one = Complex.One;
                var zero = Complex.Zero;
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    &one,
                    pu, ldu,
                    psmat, m,
                    &zero,
                    pus, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    &one,
                    pus, m,
                    pvt, ldvt,
                    &zero,
                    preconstructed, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    m, m, m,
                    &one,
                    pu, ldu,
                    pu, ldu,
                    &zero,
                    pidentity1, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    n, n, n,
                    &one,
                    pvt, ldvt,
                    pvt, ldvt,
                    &zero,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i].Real, Is.EqualTo(uCopy[i].Real).Within(0));
                    Assert.That(u[i].Imaginary, Is.EqualTo(uCopy[i].Imaginary).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i].Real, Is.EqualTo(vtCopy[i].Real).Within(0));
                    Assert.That(vt[i].Imaginary, Is.EqualTo(vtCopy[i].Imaginary).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdComplex_VectorOption(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomComplex(42, m, n, lda);

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomComplex(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomComplex(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new Complex[m * n];
            var us = new Complex[m * n];
            var reconstructed = new Complex[m * n];
            var identity1 = new Complex[m * m];
            var identity2 = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (double* ps = s)
            fixed (Complex* pu = u)
            fixed (Complex* pvt = vt)
            fixed (Complex* psmat = smat)
            fixed (Complex* pus = us)
            fixed (Complex* preconstructed = reconstructed)
            fixed (Complex* pidentity1 = identity1)
            fixed (Complex* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, null, 0, null, 0);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                original.CopyTo(a, 0);
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, null, 0);

                original.CopyTo(a, 0);
                Factorization.Svd(m, n, pa, lda, ps, null, 0, pvt, ldvt);

                var one = Complex.One;
                var zero = Complex.Zero;
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    &one,
                    pu, ldu,
                    psmat, m,
                    &zero,
                    pus, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    &one,
                    pus, m,
                    pvt, ldvt,
                    &zero,
                    preconstructed, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    m, m, m,
                    &one,
                    pu, ldu,
                    pu, ldu,
                    &zero,
                    pidentity1, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    n, n, n,
                    &one,
                    pvt, ldvt,
                    pvt, ldvt,
                    &zero,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i].Real, Is.EqualTo(uCopy[i].Real).Within(0));
                    Assert.That(u[i].Imaginary, Is.EqualTo(uCopy[i].Imaginary).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i].Real, Is.EqualTo(vtCopy[i].Real).Within(0));
                    Assert.That(vt[i].Imaginary, Is.EqualTo(vtCopy[i].Imaginary).Within(0));
                }
            }
        }

        [TestCase(1, 1, 1, 1, 1)]
        [TestCase(1, 1, 3, 2, 4)]
        [TestCase(2, 2, 2, 2, 2)]
        [TestCase(2, 2, 5, 3, 4)]
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(3, 3, 5, 4, 6)]
        [TestCase(3, 1, 3, 3, 1)]
        [TestCase(3, 1, 4, 5, 2)]
        [TestCase(1, 3, 1, 1, 3)]
        [TestCase(1, 3, 4, 2, 5)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(4, 3, 6, 5, 4)]
        [TestCase(3, 4, 3, 3, 4)]
        [TestCase(3, 4, 5, 4, 7)]
        [TestCase(11, 23, 11, 11, 23)]
        [TestCase(11, 23, 17, 19, 31)]
        [TestCase(23, 11, 23, 23, 11)]
        [TestCase(23, 11, 31, 29, 17)]
        [TestCase(16, 8, 16, 16, 8)]
        [TestCase(16, 8, 32, 32, 16)]
        [TestCase(8, 16, 8, 8, 16)]
        [TestCase(8, 16, 16, 16, 32)]
        public unsafe void SvdComplex_Singular(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomComplex(42, m, n, lda);
            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    if (row >= m / 2 || col >= n / 2)
                    {
                        Matrix.Set(m, n, original, lda, row, col, 0);
                    }
                }
            }

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomComplex(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomComplex(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new Complex[m * n];
            var us = new Complex[m * n];
            var reconstructed = new Complex[m * n];
            var identity1 = new Complex[m * m];
            var identity2 = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (double* ps = s)
            fixed (Complex* pu = u)
            fixed (Complex* pvt = vt)
            fixed (Complex* psmat = smat)
            fixed (Complex* pus = us)
            fixed (Complex* preconstructed = reconstructed)
            fixed (Complex* pidentity1 = identity1)
            fixed (Complex* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                var one = Complex.One;
                var zero = Complex.Zero;
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    &one,
                    pu, ldu,
                    psmat, m,
                    &zero,
                    pus, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    &one,
                    pus, m,
                    pvt, ldvt,
                    &zero,
                    preconstructed, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    m, m, m,
                    &one,
                    pu, ldu,
                    pu, ldu,
                    &zero,
                    pidentity1, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    n, n, n,
                    &one,
                    pvt, ldvt,
                    pvt, ldvt,
                    &zero,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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

            for (var i = 0; i < u.Length; i++)
            {
                var row = i % ldu;
                var col = i / ldu;
                if (row >= m)
                {
                    Assert.That(u[i].Real, Is.EqualTo(uCopy[i].Real).Within(0));
                    Assert.That(u[i].Imaginary, Is.EqualTo(uCopy[i].Imaginary).Within(0));
                }
            }

            for (var i = 0; i < vt.Length; i++)
            {
                var row = i % ldvt;
                var col = i / ldvt;
                if (row >= n)
                {
                    Assert.That(vt[i].Real, Is.EqualTo(vtCopy[i].Real).Within(0));
                    Assert.That(vt[i].Imaginary, Is.EqualTo(vtCopy[i].Imaginary).Within(0));
                }
            }
        }

        [Test]
        public unsafe void SvdComplex_Zero()
        {
            var n = 3;
            var m = 3;
            var lda = 3;
            var ldu = 3;
            var ldvt = 3;
            var original = new Complex[m * n];

            var a = original.ToArray();
            var s = new double[Math.Min(m, n)];
            var u = Matrix.RandomComplex(0, m, m, ldu);
            var uCopy = u.ToArray();
            var vt = Matrix.RandomComplex(0, n, n, ldvt);
            var vtCopy = vt.ToArray();
            var smat = new Complex[m * n];
            var us = new Complex[m * n];
            var reconstructed = new Complex[m * n];
            var identity1 = new Complex[m * m];
            var identity2 = new Complex[n * n];
            fixed (Complex* pa = a)
            fixed (double* ps = s)
            fixed (Complex* pu = u)
            fixed (Complex* pvt = vt)
            fixed (Complex* psmat = smat)
            fixed (Complex* pus = us)
            fixed (Complex* preconstructed = reconstructed)
            fixed (Complex* pidentity1 = identity1)
            fixed (Complex* pidentity2 = identity2)
            {
                Factorization.Svd(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
                for (var i = 0; i < s.Length; i++)
                {
                    Matrix.Set(m, n, smat, m, i, i, s[i]);
                }

                var one = Complex.One;
                var zero = Complex.Zero;
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, m,
                    &one,
                    pu, ldu,
                    psmat, m,
                    &zero,
                    pus, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.NoTrans, Transpose.NoTrans,
                    m, n, n,
                    &one,
                    pus, m,
                    pvt, ldvt,
                    &zero,
                    preconstructed, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    m, m, m,
                    &one,
                    pu, ldu,
                    pu, ldu,
                    &zero,
                    pidentity1, m);
                Blas.Zgemm(
                    Order.ColMajor,
                    Transpose.ConjTrans, Transpose.NoTrans,
                    n, n, n,
                    &one,
                    pvt, ldvt,
                    pvt, ldvt,
                    &zero,
                    pidentity2, n);
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

            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < m; col++)
                {
                    var value = identity1[col * m + row];
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

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    var value = identity2[col * n + row];
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
