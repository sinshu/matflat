using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        public static unsafe void Gevd(int n, double* a, int lda, double* b, int ldb, double* w)
        {
            var c = ArrayPool<double>.Shared.Rent(n * n);
            try
            {
                fixed (double* pc = c)
                {
                    GevdCore(n, a, lda, b, ldb, w, pc);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(c);
            }
        }

        private static unsafe void GevdCore(int n, double* a, int lda, double* b, int ldb, double* w, double* c)
        {
            // This implementation is based on the following method:
            // https://www.netlib.org/lapack/lug/node54.html

            // Cholesky B.
            Cholesky(n, b, ldb);

            // Compute C.
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var cColj = c + n * j;
                var cRowj = c + j;
                for (var i = 0; i < j; i++)
                {
                    cColj[i] = aColj[i];
                    *cRowj = aColj[i];
                    cRowj += n;
                }
                cColj[j] = aColj[j];
            }
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + i, n);
            }
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + n * i, 1);
            }

            // Solve the eigenvalue problem of C.
            Svd(n, n, c, n, w, a, lda, null, 0);

            // Recover the eigenvectors.
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.Trans, n, b, ldb, a + lda * i, 1);
            }
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var norm = Internals.Norm(n, aColj);
                Internals.DivInplace(n, aColj, norm);
            }
        }
    }
}
