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
        [TestCase(3, 3, 3)]
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

            Matrix.Print(m, n, expectedA, lda);
            Console.WriteLine();

            Matrix.Print(m, n, actualA, lda);
            Console.WriteLine();
        }
    }
}
