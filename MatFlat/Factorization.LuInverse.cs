using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        /// <summary>
        /// Computes the inverse matrix from the result of an LU decomposition.
        /// </summary>
        /// <param name="n">
        /// The order of the matrix.
        /// </param>
        /// <param name="a">
        /// The result of the LU decomposition.
        /// </param>
        /// <param name="lda">
        /// The leading dimension.
        /// </param>
        /// <param name="piv">
        /// The pivot indices obtained from the LU decomposition.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The matrix is not invertible.
        /// </exception>
        public static unsafe void LuInverse(int n, float* a, int lda, int* piv)
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

            if (piv == null)
            {
                throw new ArgumentNullException(nameof(piv));
            }

            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                if (aColj[j] == 0.0)
                {
                    throw new MatrixFactorizationException("The matrix is not invertible.");
                }
            }

            var work = ArrayPool<float>.Shared.Rent(n * n + n);
            try
            {
                fixed (float* pwork = work)
                {
                    var diag = pwork + n * n;
                    for (var j = 0; j < n; j++)
                    {
                        var aColj = a + lda * j;
                        var copySize = sizeof(float) * n;
                        Buffer.MemoryCopy(aColj, pwork + n * j, copySize, copySize);
                        diag[j] = aColj[j];
                        new Span<float>(aColj, n).Clear();
                    }

                    for (var j = 0; j < n; j++)
                    {
                        a[lda * piv[j] + j] = 1.0F;
                    }

                    for (var j = 0; j < n; j++)
                    {
                        pwork[n * j + j] = 1.0F;
                    }
                    for (var j = 0; j < n; j++)
                    {
                        var aColj = a + lda * j;
                        Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, pwork, n, aColj, 1);
                    }

                    for (var j = 0; j < n; j++)
                    {
                        pwork[n * j + j] = diag[j];
                    }
                    for (var j = 0; j < n; j++)
                    {
                        var aColj = a + lda * j;
                        Blas.SolveTriangular(Uplo.Upper, Transpose.NoTrans, n, pwork, n, aColj, 1);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(work);
            }
        }

        /// <summary>
        /// Computes the inverse matrix from the result of an LU decomposition.
        /// </summary>
        /// <param name="n">
        /// The order of the matrix.
        /// </param>
        /// <param name="a">
        /// The result of the LU decomposition.
        /// </param>
        /// <param name="lda">
        /// The leading dimension.
        /// </param>
        /// <param name="piv">
        /// The pivot indices obtained from the LU decomposition.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The matrix is not invertible.
        /// </exception>
        public static unsafe void LuInverse(int n, double* a, int lda, int* piv)
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

            if (piv == null)
            {
                throw new ArgumentNullException(nameof(piv));
            }

            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                if (aColj[j] == 0.0)
                {
                    throw new MatrixFactorizationException("The matrix is not invertible.");
                }
            }

            var work = ArrayPool<double>.Shared.Rent(n * n + n);
            try
            {
                fixed (double* pwork = work)
                {
                    var diag = pwork + n * n;
                    for (var j = 0; j < n; j++)
                    {
                        var aColj = a + lda * j;
                        var copySize = sizeof(double) * n;
                        Buffer.MemoryCopy(aColj, pwork + n * j, copySize, copySize);
                        diag[j] = aColj[j];
                        new Span<double>(aColj, n).Clear();
                    }

                    for (var j = 0; j < n; j++)
                    {
                        a[lda * piv[j] + j] = 1.0;
                    }

                    for (var j = 0; j < n; j++)
                    {
                        pwork[n * j + j] = 1.0;
                    }
                    for (var j = 0; j < n; j++)
                    {
                        var aColj = a + lda * j;
                        Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, pwork, n, aColj, 1);
                    }

                    for (var j = 0; j < n; j++)
                    {
                        pwork[n * j + j] = diag[j];
                    }
                    for (var j = 0; j < n; j++)
                    {
                        var aColj = a + lda * j;
                        Blas.SolveTriangular(Uplo.Upper, Transpose.NoTrans, n, pwork, n, aColj, 1);
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(work);
            }
        }
    }
}
