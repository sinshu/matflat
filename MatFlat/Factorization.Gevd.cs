using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        public static unsafe void Gevd(int n, double* a, int lda, double* b, int ldb, double* w)
        {
            var l = new double[n * n];
            fixed (double* pl = l)
            {
                for (var j = 0; j < n; j++)
                {
                    var copyLength = sizeof(double) * n;
                    Buffer.MemoryCopy(b + ldb * j, pl + n * j, copyLength, copyLength);
                }

                Cholesky(n, pl, n);
            }

            var c = new double[n * n];
            var s = new double[n];
            fixed (double* pc = c)
            fixed (double* pl = l)
            fixed (double* ps = s)
            {
                for (var j = 0; j < n; j++)
                {
                    var copyLength = sizeof(double) * n;
                    Buffer.MemoryCopy(a + lda * j, pc + n * j, copyLength, copyLength);
                }

                for (var i = 0; i < n; i++)
                {
                    Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, pl, n, pc + i, n);
                }

                for (var i = 0; i < n; i++)
                {
                    Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, pl, n, pc + n * i, 1);
                }

                Svd(n, n, pc, n, ps, null, 0, null, 0);
            }

            Print(n, n, l, n);
            Console.WriteLine();

            Print(n, n, c, n);
            Console.WriteLine();

            foreach (var value in s)
            {
                Console.WriteLine(value);
            }
        }

        public static T Get<T>(int m, int n, T[] a, int lda, int row, int col)
        {
            var index = col * lda + row;
            return a[index];
        }

        public static T Set<T>(int m, int n, T[] a, int lda, int row, int col, T value)
        {
            var index = col * lda + row;
            return a[index] = value;
        }

        public static void Print<T>(int m, int n, T[] a, int lda) where T : IFormattable
        {
            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    Console.Write("\t");
                    Console.Write(Get(m, n, a, lda, row, col).ToString("G6", null));
                }
                Console.WriteLine();
            }
        }
    }
}
