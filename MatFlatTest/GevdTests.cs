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
            var w = new double[n];

            Matrix.Print(n, n, a, lda);
            Console.WriteLine();

            Matrix.Print(n, n, b, ldb);
            Console.WriteLine();

            fixed (double* pa = a)
            fixed (double* pb = b)
            fixed (double* pw = w)
            {
                Factorization.Gevd(n, pa, lda, pb, ldb, pw);
            }

            foreach (var value in w)
            {
                Console.WriteLine(value);
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
                    if (col >= row)
                    {
                        var value = symmetric[n * col + row];
                        a[lda * col + row] = value;
                    }
                    else
                    {
                        a[lda * col + row] = double.NaN;
                    }
                }
            }

            return a;
        }
    }
}
