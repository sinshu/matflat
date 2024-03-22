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
                var p = incx * i;
                x[p] = (x[p] - Internals.Dot(i, a + i, lda, x, incx)) / a[lda * i + i];
            }
        }
    }
}
