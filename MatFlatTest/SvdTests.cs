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
        [TestCase(3, 3, 3, 3, 3)]
        [TestCase(4, 3, 4, 4, 3)]
        [TestCase(3, 4, 3, 3, 4)]
        public unsafe void SvdComplex_General(int m, int n, int lda, int ldu, int ldvt)
        {
            var a = Matrix.RandomComplex(42, m, n, lda);

            var expectedA = a.ToArray();
            var expectedS = new double[Math.Min(m, n)];
            var expectedU = Matrix.RandomComplex(0, m, m, ldu);
            var expectedVT = Matrix.RandomComplex(0, n, n, ldvt);
            var work = new double[Math.Min(m, n)];
            fixed (Complex* pa = expectedA)
            fixed (double* ps = expectedS)
            fixed (Complex* pu = expectedU)
            fixed (Complex* pvt = expectedVT)
            fixed (double* pwork = work)
            {
                Lapack.Zgesvd(
                    MatrixLayout.ColMajor,
                    'A', 'A',
                    m, n,
                    pa, lda,
                    ps,
                    pu, ldu,
                    pvt, ldvt,
                    pwork);
            }

            var actualA = a.ToArray();
            var actualS = new Complex[Math.Min(m, n)];
            var actualU = Matrix.RandomComplex(0, m, m, ldu);
            var actualVT = Matrix.RandomComplex(0, n, n, ldvt);
            fixed (Complex* pa = actualA)
            fixed (Complex* ps = actualS)
            fixed (Complex* pu = actualU)
            fixed (Complex* pvt = actualVT)
            {
                Factorization.SvdComplex(m, n, pa, lda, ps, pu, ldu, pvt, ldvt);
            }

            Matrix.Print(m, m, expectedU, ldu);
            Console.WriteLine();

            Matrix.Print(m, m, actualU, ldu);
            Console.WriteLine();
        }
    }
}
