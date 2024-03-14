using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class MatrixDecomposition
    {
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
                            Swap(n, a + p, a + j, lda);
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

        private static unsafe void Swap<T>(int n, T* x, T* y, int inc) where T : unmanaged, INumberBase<T>
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
    }
}
