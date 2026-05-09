using System;
using System.Numerics;
using ILNumerics;
using ILNumerics.Core.Native;
using ILNumerics.F2NET;
using MatFlat;

namespace MatFlatTest
{
    internal static class LapackTest
    {
        private static readonly ILapack lapack = new ManagedLAPACK();

        public static unsafe void Sgemm(Transpose transA, Transpose transB, int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float beta, float* c, int ldc)
        {
            lapack.sgemm(ToLapackTranspose(transA), ToLapackTranspose(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public static unsafe void Dgemm(Transpose transA, Transpose transB, int m, int n, int k, double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc)
        {
            lapack.dgemm(ToLapackTranspose(transA), ToLapackTranspose(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        public static unsafe void Zgemm(Transpose transA, Transpose transB, int m, int n, int k, Complex* alpha, Complex* a, int lda, Complex* b, int ldb, Complex* beta, Complex* c, int ldc)
        {
            lapack.zgemm(ToLapackTranspose(transA), ToLapackTranspose(transB), m, n, k, ToLapackComplex(*alpha), (complex*)a, lda, (complex*)b, ldb, ToLapackComplex(*beta), (complex*)c, ldc);
        }

        public static unsafe void Sgemv(Transpose transA, int m, int n, float alpha, float* a, int lda, float* x, int incx, float beta, float* y, int incy)
        {
            if (incx != 1 || incy != 1)
            {
                throw new NotSupportedException("ManagedLAPACK-based gemv test helper only supports unit increments.");
            }

            var rows = transA == Transpose.NoTrans ? m : n;
            var k = transA == Transpose.NoTrans ? n : m;
            lapack.sgemm(ToLapackTranspose(transA), 'N', rows, 1, k, alpha, a, lda, x, k, beta, y, rows);
        }

        public static unsafe void Dgemv(Transpose transA, int m, int n, double alpha, double* a, int lda, double* x, int incx, double beta, double* y, int incy)
        {
            if (incx != 1 || incy != 1)
            {
                throw new NotSupportedException("ManagedLAPACK-based gemv test helper only supports unit increments.");
            }

            var rows = transA == Transpose.NoTrans ? m : n;
            var k = transA == Transpose.NoTrans ? n : m;
            lapack.dgemm(ToLapackTranspose(transA), 'N', rows, 1, k, alpha, a, lda, x, k, beta, y, rows);
        }

        public static unsafe void Zgemv(Transpose transA, int m, int n, Complex* alpha, Complex* a, int lda, Complex* x, int incx, Complex* beta, Complex* y, int incy)
        {
            if (incx != 1 || incy != 1)
            {
                throw new NotSupportedException("ManagedLAPACK-based gemv test helper only supports unit increments.");
            }

            var rows = transA == Transpose.NoTrans ? m : n;
            var k = transA == Transpose.NoTrans ? n : m;
            lapack.zgemm(ToLapackTranspose(transA), 'N', rows, 1, k, ToLapackComplex(*alpha), (complex*)a, lda, (complex*)x, k, ToLapackComplex(*beta), (complex*)y, rows);
        }

        public static unsafe int Sgetrf(int m, int n, float* a, int lda, int* piv)
        {
            var info = 0;
            lapack.sgetrf(m, n, a, lda, piv, ref info);
            return info;
        }

        public static unsafe int Dgetrf(int m, int n, double* a, int lda, int* piv)
        {
            var info = 0;
            lapack.dgetrf(m, n, a, lda, piv, ref info);
            return info;
        }

        public static unsafe int Zgetrf(int m, int n, Complex* a, int lda, int* piv)
        {
            var info = 0;
            lapack.zgetrf(m, n, (complex*)a, lda, piv, ref info);
            return info;
        }

        public static unsafe int Spotrf(char uplo, int n, float* a, int lda)
        {
            var info = 0;
            lapack.spotrf(uplo, n, a, lda, ref info);
            return info;
        }

        public static unsafe int Dpotrf(char uplo, int n, double* a, int lda)
        {
            var info = 0;
            lapack.dpotrf(uplo, n, a, lda, ref info);
            return info;
        }

        public static unsafe int Zpotrf(char uplo, int n, Complex* a, int lda)
        {
            var info = 0;
            lapack.zpotrf(uplo, n, (complex*)a, lda, ref info);
            return info;
        }

        private static char ToLapackTranspose(Transpose trans)
        {
            return trans switch
            {
                Transpose.NoTrans => 'N',
                Transpose.Trans => 'T',
                Transpose.ConjTrans => 'C',
                _ => throw new ArgumentOutOfRangeException(nameof(trans), trans, null),
            };
        }

        private static complex ToLapackComplex(Complex value)
        {
            return new complex(value.Real, value.Imaginary);
        }
    }
}
