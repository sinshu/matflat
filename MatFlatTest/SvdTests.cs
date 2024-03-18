using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;
using System.Security.Principal;

namespace MatFlatTest
{
    public class SvdTests
    {
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(3, 4, 3, 3, 4)]
        public unsafe void SvdComplex_General(int m, int n, int lda, int ldu, int ldvt)
        {
            var original = Matrix.RandomComplex(42, m, n, lda);

            var a = original.ToArray();
            var s = new Complex[Math.Min(m, n)];
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
            fixed (Complex* ps = s)
            fixed (Complex* pu = u)
            fixed (Complex* pvt = vt)
            fixed (Complex* psmat = smat)
            fixed (Complex* pus = us)
            fixed (Complex* preconstructed = reconstructed)
            fixed (Complex* pidentity1 = identity1)
            fixed (Complex* pidentity2 = identity2)
            {
                Factorization.SvdComplex(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
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
    }
}
