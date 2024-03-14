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
                    // Outer loop.
                    var colj = a;
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
                        if (j < m && colj[j] != 0.0)
                        {
                            for (var i = j + 1; i < m; i++)
                            {
                                colj[i] /= colj[j];
                            }
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
            var sum1 = 0.0;
            var sum2 = 0.0;
            var ix1 = 0;
            var iy1 = 0;
            var ix2 = incx;
            var iy2 = incy;
            var incx2 = 2 * incx;
            var incy2 = 2 * incy;

            while (rem >= 2)
            {
                sum1 += x[ix1] * y[iy1];
                sum2 += x[ix2] * y[iy2];
                ix1 += incx2;
                iy1 += incy2;
                ix2 += incx2;
                iy2 += incy2;
                rem -= 2;
            }

            if (rem == 1)
            {
                sum1 += x[ix1] * y[iy1];
            }

            return sum1 + sum2;
        }
    }
}
