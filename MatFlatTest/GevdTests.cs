using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class GevdTests
    {
        [TestCase(3, 4, 5)]
        public unsafe void GevdDouble(int n, int lda, int ldb)
        {
            var a = GetDecomposableDouble(42, n, lda);
            var b = GetDecomposableDouble(57, n, ldb);

            Matrix.Print(n, n, a, lda);
            Console.WriteLine();

            Matrix.Print(n, n, b, ldb);
            Console.WriteLine();

            fixed (double* pa = a)
            fixed (double* pb = b)
            {
                Factorization.Gevd(n, pa, lda, pb, ldb, null);
            }
        }

        private static unsafe double[] GetDecomposableDouble(int seed, int n, int lda)
        {
            var a = Matrix.RandomDouble(seed, n, n, lda);

            var symmetric = new double[n * n];
            fixed (double* pa = a)
            fixed (double* ps = symmetric)
            {
                OpenBlasSharp.Blas.Dgemm(
                    Order.ColMajor,
                    OpenBlasSharp.Transpose.NoTrans,
                    OpenBlasSharp.Transpose.Trans,
                    n, n, n,
                    1.0,
                    pa, lda,
                    pa, lda,
                    0.0,
                    ps, n);
            }

            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    a[lda * col + row] = symmetric[n * col + row];
                }
            }

            return a;
        }
    }
}
