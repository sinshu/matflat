﻿using System;

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
                        for (var j = i - 1; j >= 0; j--)
                        {
                            x[j * incx] -= aColi[j] * x[p];
                        }
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
                        for (var i = j - 1; i >= 0; i--)
                        {
                            x[i * incx] -= a[i * lda + j] * x[p];
                        }
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
    }
}