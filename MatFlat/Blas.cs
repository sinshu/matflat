using System;

namespace MatFlat
{
    /// <summary>
    /// Provides a subset of the BLAS routines.
    /// </summary>
    public static class Blas
    {
        public static unsafe void SolveTriangular(Uplo uplo, Transpose transa, int n, double* a, int lda, double* x, int incx)
        {
            if (uplo == Uplo.Upper)
            {
                if (transa == Transpose.NoTrans)
                {
                }
                else if (transa == Transpose.Trans)
                {
                    for (var j = 0; j < n; j++)
                    {
                        var p = incx * j;
                        x[p] = (x[p] - Internals.Dot(j, a + lda * j, 1, x, incx)) / a[lda * j + j];
                    }
                }
                else
                {
                    throw new ArgumentException("Invalid enum value.", nameof(transa));
                }
            }
            else if (uplo == Uplo.Lower)
            {
                if (transa == Transpose.NoTrans)
                {
                    for (var i = 0; i < n; i++)
                    {
                        var p = incx * i;
                        x[p] = (x[p] - Internals.Dot(i, a + i, lda, x, incx)) / a[lda * i + i];
                    }
                }
                else if (transa == Transpose.Trans)
                {
                }
                else
                {
                    throw new ArgumentException("Invalid enum value.", nameof(transa));
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(uplo));
            }
        }
    }
}
