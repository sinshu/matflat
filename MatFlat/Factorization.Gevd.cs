using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        /// <summary>
        /// Computes the generalized eigenvalue decomposition (GEVD) of
        /// an N-by-N Hermitian matrix A and a positive definite matrix B.
        /// </summary>
        /// <param name="n">
        /// The order of the matrices A and B.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the Hermitian matrix A.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of A contains
        /// the upper triangular part of the matrix A, and
        /// the lower triangular part of A is not referenced.
        /// </para>
        /// <para>
        /// On exit, it contains the eigenvectors.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="b">
        /// <para>
        /// On entry, the Hermitian positive definite matrix B.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of B contains
        /// the upper triangular part of the matrix B, and
        /// the lower triangular part of B is not referenced.
        /// </para>
        /// <para>
        /// On exit, the factor L from the Cholesky factorization <c>B = L * L^H</c>.
        /// </para>
        /// </param>
        /// <param name="ldb">
        /// The leading dimension of the array B.
        /// </param>
        /// <param name="w">
        /// On exit, it contains the eigenvalues.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The GEVD failed. This can happen if B is not positive definite.
        /// </exception>
        public static unsafe void Gevd(int n, float* a, int lda, float* b, int ldb, float* w)
        {
            if (n <= 0)
            {
                throw new ArgumentException("The order of the matrix must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(lda));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (ldb < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(ldb));
            }

            if (w == null)
            {
                throw new ArgumentNullException(nameof(w));
            }

            var c = ArrayPool<float>.Shared.Rent(n * n);
            try
            {
                fixed (float* pc = c)
                {
                    GevdCore(n, a, lda, b, ldb, w, pc);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(c);
            }
        }

        /// <summary>
        /// Computes the generalized eigenvalue decomposition (GEVD) of
        /// an N-by-N Hermitian matrix A and a positive definite matrix B.
        /// </summary>
        /// <param name="n">
        /// The order of the matrices A and B.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the Hermitian matrix A.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of A contains
        /// the upper triangular part of the matrix A, and
        /// the lower triangular part of A is not referenced.
        /// </para>
        /// <para>
        /// On exit, it contains the eigenvectors.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="b">
        /// <para>
        /// On entry, the Hermitian positive definite matrix B.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of B contains
        /// the upper triangular part of the matrix B, and
        /// the lower triangular part of B is not referenced.
        /// </para>
        /// <para>
        /// On exit, the factor L from the Cholesky factorization <c>B = L * L^H</c>.
        /// </para>
        /// </param>
        /// <param name="ldb">
        /// The leading dimension of the array B.
        /// </param>
        /// <param name="w">
        /// On exit, it contains the eigenvalues.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The GEVD failed. This can happen if B is not positive definite.
        /// </exception>
        public static unsafe void Gevd(int n, double* a, int lda, double* b, int ldb, double* w)
        {
            if (n <= 0)
            {
                throw new ArgumentException("The order of the matrix must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(lda));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (ldb < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(ldb));
            }

            if (w == null)
            {
                throw new ArgumentNullException(nameof(w));
            }

            var c = ArrayPool<double>.Shared.Rent(n * n);
            try
            {
                fixed (double* pc = c)
                {
                    GevdCore(n, a, lda, b, ldb, w, pc);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(c);
            }
        }

        /// <summary>
        /// Computes the generalized eigenvalue decomposition (GEVD) of
        /// an N-by-N Hermitian matrix A and a positive definite matrix B.
        /// </summary>
        /// <param name="n">
        /// The order of the matrices A and B.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the Hermitian matrix A.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of A contains
        /// the upper triangular part of the matrix A, and
        /// the lower triangular part of A is not referenced.
        /// </para>
        /// <para>
        /// On exit, it contains the eigenvectors.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="b">
        /// <para>
        /// On entry, the Hermitian positive definite matrix B.
        /// </para>
        /// <para>
        /// The leading N-by-N upper triangular part of B contains
        /// the upper triangular part of the matrix B, and
        /// the lower triangular part of B is not referenced.
        /// </para>
        /// <para>
        /// On exit, the factor L from the Cholesky factorization <c>B = L * L^H</c>.
        /// </para>
        /// </param>
        /// <param name="ldb">
        /// The leading dimension of the array B.
        /// </param>
        /// <param name="w">
        /// On exit, it contains the eigenvalues.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The GEVD failed. This can happen if B is not positive definite.
        /// </exception>
        public static unsafe void Gevd(int n, Complex* a, int lda, Complex* b, int ldb, double* w)
        {
            if (n <= 0)
            {
                throw new ArgumentException("The order of the matrix must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(lda));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (ldb < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the order of the matrix.", nameof(ldb));
            }

            if (w == null)
            {
                throw new ArgumentNullException(nameof(w));
            }

            var c = ArrayPool<Complex>.Shared.Rent(n * n);
            try
            {
                fixed (Complex* pc = c)
                {
                    GevdCore(n, a, lda, b, ldb, w, pc);
                }
            }
            finally
            {
                ArrayPool<Complex>.Shared.Return(c);
            }
        }

        private static unsafe void GevdCore(int n, float* a, int lda, float* b, int ldb, float* w, float* c)
        {
            // This implementation is based on the following method:
            // https://www.netlib.org/lapack/lug/node54.html

            // Cholesky B.
            try
            {
                Cholesky(n, b, ldb);
            }
            catch (MatFlatException)
            {
                throw new MatFlatException("The right-hand side matrix is not positive definite.");
            }

            // Compute C.
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var cColj = c + n * j;
                var cRowj = c + j;
                for (var i = 0; i < j; i++)
                {
                    cColj[i] = aColj[i];
                    *cRowj = aColj[i];
                    cRowj += n;
                }
                cColj[j] = aColj[j];
            }
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + i, n);
            }
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + n * i, 1);
            }

            // Solve the eigenvalue problem of C.
            Svd(n, n, c, n, w, a, lda, null, 0);

            // Recover the eigenvectors.
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.Trans, n, b, ldb, a + lda * i, 1);
            }

            // Normalize the eigenvectors.
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var norm = (float)Internals.Norm(n, aColj);
                Internals.DivInplace(n, aColj, norm);
            }
        }

        private static unsafe void GevdCore(int n, double* a, int lda, double* b, int ldb, double* w, double* c)
        {
            // This implementation is based on the following method:
            // https://www.netlib.org/lapack/lug/node54.html

            // Cholesky B.
            try
            {
                Cholesky(n, b, ldb);
            }
            catch (MatFlatException)
            {
                throw new MatFlatException("The right-hand side matrix is not positive definite.");
            }

            // Compute C.
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var cColj = c + n * j;
                var cRowj = c + j;
                for (var i = 0; i < j; i++)
                {
                    cColj[i] = aColj[i];
                    *cRowj = aColj[i];
                    cRowj += n;
                }
                cColj[j] = aColj[j];
            }
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + i, n);
            }
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + n * i, 1);
            }

            // Solve the eigenvalue problem of C.
            Svd(n, n, c, n, w, a, lda, null, 0);

            // Recover the eigenvectors.
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.Trans, n, b, ldb, a + lda * i, 1);
            }

            // Normalize the eigenvectors.
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var norm = Internals.Norm(n, aColj);
                Internals.DivInplace(n, aColj, norm);
            }
        }

        private static unsafe void GevdCore(int n, Complex* a, int lda, Complex* b, int ldb, double* w, Complex* c)
        {
            // This implementation is based on the following method:
            // https://www.netlib.org/lapack/lug/node54.html

            // Cholesky B.
            try
            {
                Cholesky(n, b, ldb);
            }
            catch (MatFlatException)
            {
                throw new MatFlatException("The right-hand side matrix is not positive definite.");
            }

            // Compute C.
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var cColj = c + n * j;
                var cRowj = c + j;
                for (var i = 0; i < j; i++)
                {
                    cColj[i] = aColj[i];
                    *cRowj = aColj[i].Conjugate();
                    cRowj += n;
                }
                cColj[j] = aColj[j];
            }
            for (var i = 0; i < n; i++)
            {
                for (var k = 0; k < n; k++)
                {
                    c[n * k + i] = c[n * k + i].Conjugate();
                }
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + i, n);
                for (var k = 0; k < n; k++)
                {
                    c[n * k + i] = c[n * k + i].Conjugate();
                }
            }
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.NoTrans, n, b, ldb, c + n * i, 1);
            }

            // Solve the eigenvalue problem of C.
            Svd(n, n, c, n, w, a, lda, null, 0);

            // Recover the eigenvectors.
            for (var i = 0; i < n; i++)
            {
                Blas.SolveTriangular(Uplo.Lower, Transpose.ConjTrans, n, b, ldb, a + lda * i, 1);
            }

            // Normalize the eigenvectors.
            for (var j = 0; j < n; j++)
            {
                var aColj = a + lda * j;
                var norm = Internals.Norm(n, aColj);
                Internals.DivInplace(n, aColj, norm);
            }
        }
    }
}
