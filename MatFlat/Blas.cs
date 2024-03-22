using System;

namespace MatFlat
{
    /// <summary>
    /// Provides a subset of the BLAS routines.
    /// </summary>
    public static class Blas
    {
        public static unsafe void ForwardSubstitution(int n, double* a, int lda, double* x, int incx)
        {
            for (var i = 0; i < n; i++)
            {
                var sum = x[i * incx];
                for (var j = 0; j < i; j++)
                {
                    sum -= a[j * lda + i] * x[j * incx];
                }
                x[i * incx] = sum / a[i * lda + i];
            }
        }
    }
}
