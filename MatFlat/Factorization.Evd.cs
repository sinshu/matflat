using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        /// <summary>
        /// Computes the eigenvalue decomposition (EVD) of
        /// an N-by-N Hermitian matrix A.
        /// </summary>
        /// <param name="n">
        /// The order of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the Hermitian matrix A.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of A contains
        /// the upper triangular part of the matrix A, and
        /// the lower triangular part of A is not referenced.
        /// </para>
        /// <para>
        /// On exit, it contains the eigenvectors.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="w">
        /// On exit, it contains the eigenvalues.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The solution did not converge.
        /// </exception>
        public static unsafe void Evd(int n, float* a, int lda, float* w)
        {
            if (n <= 0)
            {
                throw new ArgumentException("The order of the matrix must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(lda));
            }

            if (w == null)
            {
                throw new ArgumentNullException(nameof(w));
            }

            var c = ArrayPool<float>.Shared.Rent(n * n);
            try
            {
                fixed (float* pc = c)
                {
                    EvdCore(n, a, lda, w, pc);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(c);
            }
        }

        /// <summary>
        /// Computes the eigenvalue decomposition (EVD) of
        /// an N-by-N Hermitian matrix A.
        /// </summary>
        /// <param name="n">
        /// The order of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the Hermitian matrix A.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of A contains
        /// the upper triangular part of the matrix A, and
        /// the lower triangular part of A is not referenced.
        /// </para>
        /// <para>
        /// On exit, it contains the eigenvectors.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="w">
        /// On exit, it contains the eigenvalues.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The solution did not converge.
        /// </exception>
        public static unsafe void Evd(int n, double* a, int lda, double* w)
        {
            if (n <= 0)
            {
                throw new ArgumentException("The order of the matrix must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(lda));
            }

            if (w == null)
            {
                throw new ArgumentNullException(nameof(w));
            }

            var c = ArrayPool<double>.Shared.Rent(n * n);
            try
            {
                fixed (double* pc = c)
                {
                    EvdCore(n, a, lda, w, pc);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(c);
            }
        }

        private static unsafe void EvdCore(int n, float* a, int lda, float* w, float* c)
        {
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

            Svd(n, n, c, n, w, a, lda, null, 0);
        }

        private static unsafe void EvdCore(int n, double* a, int lda, double* w, double* c)
        {
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

            Svd(n, n, c, n, w, a, lda, null, 0);
        }
    }
}
