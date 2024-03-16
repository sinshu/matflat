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
        public static unsafe void LuSingle(int m, int n, float* a, int lda, int* piv)
        {
            // Initialize the pivot matrix to the identity permutation.
            for (var i = 0; i < m; i++)
            {
                piv[i] = i;
            }

            var buffer = ArrayPool<float>.Shared.Rent(m);
            try
            {
                fixed (float* luColj = buffer)
                {
                    var colj = a;

                    // Outer loop.
                    for (var j = 0; j < n; j++)
                    {
                        // Make a copy of the j-th column to localize references.
                        var copySize = sizeof(float) * m;
                        Buffer.MemoryCopy(colj, luColj, copySize, copySize);

                        // Apply previous transformations.
                        for (var i = 0; i < m; i++)
                        {
                            // Most of the time is spent in the following dot product.
                            var s = LuDot(Math.Min(i, j), a + i, lda, luColj);
                            colj[i] = luColj[i] -= s;
                        }

                        // Find pivot and exchange if necessary.
                        var p = j;
                        for (var i = j + 1; i < m; i++)
                        {
                            if (Math.Abs(luColj[i]) > Math.Abs(luColj[p]))
                            {
                                p = i;
                            }
                        }

                        if (p != j)
                        {
                            LuSwapRows(n, a + p, a + j, lda);
                            piv[j] = p;
                        }

                        // Compute multipliers.
                        var diag = colj[j];
                        if (j < m && diag != 0.0F)
                        {
                            LuDivInplace(m - j - 1, colj + j + 1, diag);
                        }

                        colj += lda;
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
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
        public static unsafe void LuDouble(int m, int n, double* a, int lda, int* piv)
        {
            // Initialize the pivot matrix to the identity permutation.
            for (var i = 0; i < m; i++)
            {
                piv[i] = i;
            }

            var buffer = ArrayPool<double>.Shared.Rent(m);
            try
            {
                fixed (double* luColj = buffer)
                {
                    var colj = a;

                    // Outer loop.
                    for (var j = 0; j < n; j++)
                    {
                        // Make a copy of the j-th column to localize references.
                        var copySize = sizeof(double) * m;
                        Buffer.MemoryCopy(colj, luColj, copySize, copySize);

                        // Apply previous transformations.
                        for (var i = 0; i < m; i++)
                        {
                            // Most of the time is spent in the following dot product.
                            var s = LuDot(Math.Min(i, j), a + i, lda, luColj);
                            colj[i] = luColj[i] -= s;
                        }

                        // Find pivot and exchange if necessary.
                        var p = j;
                        for (var i = j + 1; i < m; i++)
                        {
                            if (Math.Abs(luColj[i]) > Math.Abs(luColj[p]))
                            {
                                p = i;
                            }
                        }

                        if (p != j)
                        {
                            LuSwapRows(n, a + p, a + j, lda);
                            piv[j] = p;
                        }

                        // Compute multipliers.
                        var diag = colj[j];
                        if (j < m && diag != 0.0)
                        {
                            LuDivInplace(m - j - 1, colj + j + 1, diag);
                        }

                        colj += lda;
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(buffer);
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
        public static unsafe void LuComplex(int m, int n, Complex* a, int lda, int* piv)
        {
            // Initialize the pivot matrix to the identity permutation.
            for (var i = 0; i < m; i++)
            {
                piv[i] = i;
            }

            var buffer = ArrayPool<Complex>.Shared.Rent(m);
            try
            {
                fixed (Complex* luColj = buffer)
                {
                    var colj = a;

                    // Outer loop.
                    for (var j = 0; j < n; j++)
                    {
                        // Make a copy of the j-th column to localize references.
                        var copySize = sizeof(Complex) * m;
                        Buffer.MemoryCopy(colj, luColj, copySize, copySize);

                        // Apply previous transformations.
                        for (var i = 0; i < m; i++)
                        {
                            // Most of the time is spent in the following dot product.
                            var s = LuDot(Math.Min(i, j), a + i, lda, luColj);
                            colj[i] = luColj[i] -= s;
                        }

                        // Find pivot and exchange if necessary.
                        var p = j;
                        for (var i = j + 1; i < m; i++)
                        {
                            if (LuFastMagnitude(luColj[i]) > LuFastMagnitude(luColj[p]))
                            {
                                p = i;
                            }
                        }

                        if (p != j)
                        {
                            LuSwapRows(n, a + p, a + j, lda);
                            piv[j] = p;
                        }

                        // Compute multipliers.
                        var diag = colj[j];
                        if (j < m && diag != Complex.Zero)
                        {
                            LuDivInplace(m - j - 1, colj + j + 1, diag);
                        }

                        colj += lda;
                    }
                }
            }
            finally
            {
                ArrayPool<Complex>.Shared.Return(buffer);
            }
        }

        private static unsafe T LuDot<T>(int n, T* x, int incx, T* y) where T : unmanaged, INumberBase<T>
        {
            T sum;
            switch (n & 1)
            {
                case 0:
                    sum = T.Zero;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x += incx;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * y[0] + x[incx] * y[1];
                x += 2 * incx;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        private static unsafe void LuSwapRows<T>(int n, T* x, T* y, int inc) where T : unmanaged, INumberBase<T>
        {
            while (n > 0)
            {
                (*x, *y) = (*y, *x);
                x += inc;
                y += inc;
                n--;
            }
        }

        private static unsafe void LuDivInplace<T>(int n, T* x, T y) where T : unmanaged, INumberBase<T>
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    x[0] /= y;
                    x++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] /= y;
                x[1] /= y;
                x += 2;
                n -= 2;
            }
        }

        private static double LuFastMagnitude(Complex x)
        {
            return Math.Abs(x.Real) + Math.Abs(x.Imaginary);
        }
    }
}
