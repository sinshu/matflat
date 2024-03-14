using System;
using System.Buffers;

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
                            var kmax = Math.Min(i, j);
                            var s = Dot(kmax, a + i, lda, luColj, 1);
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
                            for (var k = 0; k < n; k++)
                            {
                                var indexk = k * lda;
                                var indexkp = indexk + p;
                                var indexkj = indexk + j;
                                (a[indexkp], a[indexkj]) = (a[indexkj], a[indexkp]);
                            }

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

        private static unsafe double Dot(int n, double* x, int incx, double* y, int incy)
        {
            var rem = n;
            var sum = 0.0;
            var ix = 0;
            var iy = 0;

            while (rem >= 2)
            {
                sum += x[ix] * y[iy] + x[ix + incx] * y[iy + incy];
                ix += 2 * incx;
                iy += 2 * incy;
                rem -= 2;
            }

            if (rem == 1)
            {
                sum += x[ix] * y[iy];
            }

            return sum;
        }

        private static unsafe void DivInplace(int n, double* x, double y)
        {
            var rem = n;
            var i = 0;

            while (rem >= 2)
            {
                x[i] /= y;
                x[i + 1] /= y;
                i += 2;
                rem -= 2;
            }

            if (rem == 1)
            {
                x[i] /= y;
            }
        }
    }
}
