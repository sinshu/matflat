using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        /// <summary>
        /// Computes an LU factorization of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the factors L and U from the factorization
        /// <c>A = P * L * U</c>.
        /// The unit diagonal elements of L are not stored.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="piv">
        /// <para>
        /// The pivot indices.
        /// </para>
        /// <para>
        /// On exit, it contains the pivot indices.
        /// </para>
        /// <para>
        /// The size of the array must be <paramref name="m"/>.
        /// </para>
        /// </param>
        /// <returns>
        /// The pivot sign.
        /// </returns>
        public static unsafe int Lu(int m, int n, float* a, int lda, int* piv)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (piv == null)
            {
                throw new ArgumentNullException(nameof(piv));
            }

            var work = ArrayPool<float>.Shared.Rent(m);
            try
            {
                fixed (float* pwork = work)
                {
                    return LuCore(m, n, a, lda, piv, pwork);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(work);
            }
        }

        /// <summary>
        /// Computes an LU factorization of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the factors L and U from the factorization
        /// <c>A = P * L * U</c>.
        /// The unit diagonal elements of L are not stored.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="piv">
        /// <para>
        /// The pivot indices.
        /// </para>
        /// <para>
        /// On exit, it contains the pivot indices.
        /// </para>
        /// <para>
        /// The size of the array must be <paramref name="m"/>.
        /// </para>
        /// </param>
        /// <returns>
        /// The pivot sign.
        /// </returns>
        public static unsafe int Lu(int m, int n, double* a, int lda, int* piv)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (piv == null)
            {
                throw new ArgumentNullException(nameof(piv));
            }

            var work = ArrayPool<double>.Shared.Rent(m);
            try
            {
                fixed (double* pwork = work)
                {
                    return LuCore(m, n, a, lda, piv, pwork);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(work);
            }
        }

        /// <summary>
        /// Computes an LU factorization of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the factors L and U from the factorization
        /// <c>A = P * L * U</c>.
        /// The unit diagonal elements of L are not stored.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="piv">
        /// <para>
        /// The pivot indices.
        /// </para>
        /// <para>
        /// On exit, it contains the pivot indices.
        /// </para>
        /// <para>
        /// The size of the array must be <paramref name="m"/>.
        /// </para>
        /// </param>
        /// <returns>
        /// The pivot sign.
        /// </returns>
        public static unsafe int Lu(int m, int n, Complex* a, int lda, int* piv)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (piv == null)
            {
                throw new ArgumentNullException(nameof(piv));
            }

            var work = ArrayPool<Complex>.Shared.Rent(m);
            try
            {
                fixed (Complex* pwork = work)
                {
                    return LuCore(m, n, a, lda, piv, pwork);
                }
            }
            finally
            {
                ArrayPool<Complex>.Shared.Return(work);
            }
        }

        private static unsafe int LuCore(int m, int n, float* a, int lda, int* piv, float* work)
        {
            // Initialize the pivot matrix to the identity permutation.
            for (var i = 0; i < m; i++)
            {
                piv[i] = i;
            }
            var sign = 1;

            var colj = a;

            // Outer loop.
            for (var j = 0; j < n; j++)
            {
                // Make a copy of the j-th column to localize references.
                var copySize = sizeof(float) * m;
                Buffer.MemoryCopy(colj, work, copySize, copySize);

                // Apply previous transformations.
                for (var i = 0; i < m; i++)
                {
                    // Most of the time is spent in the following dot product.
                    var s = (float)Internals.Dot(Math.Min(i, j), a + i, lda, work, 1);
                    colj[i] = work[i] -= s;
                }

                // Find pivot and exchange if necessary.
                var p = j;
                for (var i = j + 1; i < m; i++)
                {
                    if (Math.Abs(work[i]) > Math.Abs(work[p]))
                    {
                        p = i;
                    }
                }

                if (p != j)
                {
                    Internals.SwapRows(n, a + p, a + j, lda);
                    (piv[p], piv[j]) = (piv[j], piv[p]);
                    sign = -sign;
                }

                // Compute multipliers.
                var diag = colj[j];
                if (j < m && diag != 0.0F)
                {
                    Internals.DivInplace(m - j - 1, colj + j + 1, diag);
                }

                colj += lda;
            }

            return sign;
        }

        private static unsafe int LuCore(int m, int n, double* a, int lda, int* piv, double* work)
        {
            // Initialize the pivot matrix to the identity permutation.
            for (var i = 0; i < m; i++)
            {
                piv[i] = i;
            }
            var sign = 1;

            var colj = a;

            // Outer loop.
            for (var j = 0; j < n; j++)
            {
                // Make a copy of the j-th column to localize references.
                var copySize = sizeof(double) * m;
                Buffer.MemoryCopy(colj, work, copySize, copySize);

                // Apply previous transformations.
                for (var i = 0; i < m; i++)
                {
                    // Most of the time is spent in the following dot product.
                    var s = Internals.Dot(Math.Min(i, j), a + i, lda, work, 1);
                    colj[i] = work[i] -= s;
                }

                // Find pivot and exchange if necessary.
                var p = j;
                for (var i = j + 1; i < m; i++)
                {
                    if (Math.Abs(work[i]) > Math.Abs(work[p]))
                    {
                        p = i;
                    }
                }

                if (p != j)
                {
                    Internals.SwapRows(n, a + p, a + j, lda);
                    (piv[p], piv[j]) = (piv[j], piv[p]);
                    sign = -sign;
                }

                // Compute multipliers.
                var diag = colj[j];
                if (j < m && diag != 0.0)
                {
                    Internals.DivInplace(m - j - 1, colj + j + 1, diag);
                }

                colj += lda;
            }

            return sign;
        }

        private static unsafe int LuCore(int m, int n, Complex* a, int lda, int* piv, Complex* work)
        {
            // Initialize the pivot matrix to the identity permutation.
            for (var i = 0; i < m; i++)
            {
                piv[i] = i;
            }
            var sign = 1;

            var colj = a;

            // Outer loop.
            for (var j = 0; j < n; j++)
            {
                // Make a copy of the j-th column to localize references.
                var copySize = sizeof(Complex) * m;
                Buffer.MemoryCopy(colj, work, copySize, copySize);

                // Apply previous transformations.
                for (var i = 0; i < m; i++)
                {
                    // Most of the time is spent in the following dot product.
                    var s = Internals.Dot(Math.Min(i, j), a + i, lda, work, 1);
                    colj[i] = work[i] -= s;
                }

                // Find pivot and exchange if necessary.
                var p = j;
                for (var i = j + 1; i < m; i++)
                {
                    if (work[i].FastMagnitude() > work[p].FastMagnitude())
                    {
                        p = i;
                    }
                }

                if (p != j)
                {
                    Internals.SwapRows(n, a + p, a + j, lda);
                    (piv[p], piv[j]) = (piv[j], piv[p]);
                    sign = -sign;
                }

                // Compute multipliers.
                var diag = colj[j];
                if (j < m && diag != Complex.Zero)
                {
                    Internals.DivInplace(m - j - 1, colj + j + 1, diag);
                }

                colj += lda;
            }

            return sign;
        }
    }
}
