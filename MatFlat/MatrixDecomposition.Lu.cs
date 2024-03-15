using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class MatrixDecomposition
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
        /// <code>A = P * L * U</code>
        /// the unit diagonal elements of L are not stored.
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
                        new Span<float>(colj, m).CopyTo(new Span<float>(luColj, m));

                        // Apply previous transformations.
                        for (var i = 0; i < m; i++)
                        {
                            // Most of the time is spent in the following dot product.
                            var s = Dot(Math.Min(i, j), a + i, lda, luColj);
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
                            SwapRows(n, a + p, a + j, lda);
                            piv[j] = p;
                        }

                        // Compute multipliers.
                        var diag = colj[j];
                        if (j < m && diag != 0.0F)
                        {
                            DivInplace(m - j - 1, colj + j + 1, diag);
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
        /// <code>A = P * L * U</code>
        /// the unit diagonal elements of L are not stored.
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
                        new Span<double>(colj, m).CopyTo(new Span<double>(luColj, m));

                        // Apply previous transformations.
                        for (var i = 0; i < m; i++)
                        {
                            // Most of the time is spent in the following dot product.
                            var s = Dot(Math.Min(i, j), a + i, lda, luColj);
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
                            SwapRows(n, a + p, a + j, lda);
                            piv[j] = p;
                        }

                        // Compute multipliers.
                        var diag = colj[j];
                        if (j < m && diag != 0.0)
                        {
                            DivInplace(m - j - 1, colj + j + 1, diag);
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
        /// <code>A = P * L * U</code>
        /// the unit diagonal elements of L are not stored.
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
                        new Span<Complex>(colj, m).CopyTo(new Span<Complex>(luColj, m));

                        // Apply previous transformations.
                        for (var i = 0; i < m; i++)
                        {
                            // Most of the time is spent in the following dot product.
                            var s = Dot(Math.Min(i, j), a + i, lda, luColj);
                            colj[i] = luColj[i] -= s;
                        }

                        // Find pivot and exchange if necessary.
                        var p = j;
                        for (var i = j + 1; i < m; i++)
                        {
                            if (FastMagnitude(luColj[i]) > FastMagnitude(luColj[p]))
                            {
                                p = i;
                            }
                        }

                        if (p != j)
                        {
                            SwapRows(n, a + p, a + j, lda);
                            piv[j] = p;
                        }

                        // Compute multipliers.
                        var diag = colj[j];
                        if (j < m && diag != Complex.Zero)
                        {
                            DivInplace(m - j - 1, colj + j + 1, diag);
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

        private static unsafe T Dot<T>(int n, T* x, int incx, T* y) where T : unmanaged, INumberBase<T>
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
                    throw new Exception();
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

        private static unsafe void SwapRows<T>(int n, T* x, T* y, int inc) where T : unmanaged, INumberBase<T>
        {
            while (n > 0)
            {
                (*x, *y) = (*y, *x);
                x += inc;
                y += inc;
                n--;
            }
        }

        private static unsafe void DivInplace<T>(int n, T* x, T y) where T : unmanaged, INumberBase<T>
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
                    throw new Exception();
            }

            while (n > 0)
            {
                x[0] /= y;
                x[1] /= y;
                x += 2;
                n -= 2;
            }
        }

        private static double FastMagnitude(Complex x)
        {
            return Math.Abs(x.Real) + Math.Abs(x.Imaginary);
        }
    }
}
