using System;
using System.Numerics;

namespace MatFlat
{
    public static partial class Blas
    {
        /// <summary>
        /// <para>
        /// Solves one of the systems of equations
        /// </para>
        /// <para>
        /// <c>A * x = b</c>, or <c>A^T * x = b</c>,
        /// </para>
        /// where b and x are N-element vectors and A is an N-by-N
        /// upper or lower triangular matrix.
        /// </summary>
        /// <param name="uplo">
        /// Specifies whether the matrix is an upper or lower triangular matrix.
        /// </param>
        /// <param name="transa">
        /// Specifies whether the matrix is treated as transposed or not.
        /// </param>
        /// <param name="n">
        /// The order of the matrix A.
        /// </param>
        /// <param name="a">
        /// The matrix A.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="x">
        /// <para>
        /// On entry, it contains the N-element right-hand side vector b.
        /// </para>
        /// <para>
        /// On exit, it is overwritten with the solution vector x.
        /// </para>
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        public static unsafe void SolveTriangular(Uplo uplo, Transpose transa, int n, double* a, int lda, double* x, int incx)
        {
            if (uplo == Uplo.Upper)
            {
                if (transa == Transpose.NoTrans)
                {
                    for (var i = n - 1; i >= 0; i--)
                    {
                        var p = incx * i;
                        var aColi = a + lda * i;
                        x[p] /= aColi[i];
                        MulSub(i, aColi, 1, x[p], x, incx);
                    }
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
                    for (var j = n - 1; j >= 0; j--)
                    {
                        var p = incx * j;
                        x[p] /= a[j * lda + j];
                        MulSub(j, a + j, lda, x[p], x, incx);
                    }
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

        internal static unsafe void MulSub<T>(int n, T* x, int incx, T y, T* dst, int incdst) where T : unmanaged, INumberBase<T>
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    dst[0] -= x[0] * y;
                    x += incx;
                    dst += incdst;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                dst[0] -= x[0] * y;
                dst[incdst] -= x[incx] * y;
                x += 2 * incx;
                dst += 2 * incdst;
                n -= 2;
            }
        }
    }
}
