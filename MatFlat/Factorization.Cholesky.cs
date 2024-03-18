using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace MatFlat
{
    public static partial class Factorization
    {
        /// <summary>
        /// Computes the Cholesky factorization of a Hermitian matrix.
        /// </summary>
        /// <param name="n">
        /// The order of the matrix A.
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
        /// On exit, the factor L from the Cholesky factorization <c>A = L * L^H</c>.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <exception cref="LinearAlgebraException">
        /// The matrix is not positive definite.
        /// </exception>
        public static unsafe void CholeskySingle(int n, float* a, int lda)
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

            var colj = a;

            for (var j = 0; j < n; j++)
            {
                var s = 0.0;

                var colk = a;

                for (var k = 0; k < j; k++)
                {
                    var t = (colj[k] - CholDot(k, a + j, a + k, lda)) / colk[k];
                    colk[j] = (float)t;
                    s += t * t;

                    colk += lda;
                }

                s = colj[j] - s;

                if (s > 0)
                {
                    colj[j] = (float)Math.Sqrt(s);
                }
                else
                {
                    throw new LinearAlgebraException("Cholesky decomposition failed. The matrix must be positive definite.");
                }

                colj += lda;
            }

            colj = a + lda;

            for (var j = 1; j < n; j++)
            {
                new Span<float>(colj, j).Clear();

                colj += lda;
            }
        }

        /// <summary>
        /// Computes the Cholesky factorization of a Hermitian matrix.
        /// </summary>
        /// <param name="n">
        /// The order of the matrix A.
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
        /// On exit, the factor L from the Cholesky factorization <c>A = L * L^H</c>.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <exception cref="LinearAlgebraException">
        /// The matrix is not positive definite.
        /// </exception>
        public static unsafe void CholeskyDouble(int n, double* a, int lda)
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

            var colj = a;

            for (var j = 0; j < n; j++)
            {
                var s = 0.0;

                var colk = a;

                for (var k = 0; k < j; k++)
                {
                    var t = (colj[k] - CholDot(k, a + j, a + k, lda)) / colk[k];
                    colk[j] = t;
                    s += t * t;

                    colk += lda;
                }

                s = colj[j] - s;

                if (s > 0)
                {
                    colj[j] = Math.Sqrt(s);
                }
                else
                {
                    throw new LinearAlgebraException("Cholesky decomposition failed. The matrix must be positive definite.");
                }

                colj += lda;
            }

            colj = a + lda;

            for (var j = 1; j < n; j++)
            {
                new Span<double>(colj, j).Clear();

                colj += lda;
            }
        }

        /// <summary>
        /// Computes the Cholesky factorization of a Hermitian matrix.
        /// </summary>
        /// <param name="n">
        /// The order of the matrix A.
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
        /// On exit, the factor L from the Cholesky factorization <c>A = L * L^H</c>.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <exception cref="LinearAlgebraException">
        /// The matrix is not positive definite.
        /// </exception>
        public static unsafe void CholeskyComplex(int n, Complex* a, int lda)
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

            var colj = a;

            for (var j = 0; j < n; j++)
            {
                var s = 0.0;

                var colk = a;

                for (var k = 0; k < j; k++)
                {
                    var t = (colj[k] - CholDot(k, a + j, a + k, lda)) / colk[k];
                    colk[j] = new Complex(t.Real, -t.Imaginary);
                    s += t.Real * t.Real + t.Imaginary * t.Imaginary;

                    colk += lda;
                }

                s = colj[j].Real - s;

                if (s > 0)
                {
                    colj[j] = Math.Sqrt(s);
                }
                else
                {
                    throw new LinearAlgebraException("Cholesky decomposition failed. The matrix must be positive definite.");
                }

                colj += lda;
            }

            colj = a + lda;

            for (var j = 1; j < n; j++)
            {
                new Span<Complex>(colj, j).Clear();

                colj += lda;
            }
        }

        private static unsafe double CholDot(int n, float* x, float* y, int inc)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0;
                    break;
                case 1:
                    sum = (double)x[0] * (double)y[0];
                    x += inc;
                    y += inc;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            var inc2 = 2 * inc;
            while (n > 0)
            {
                sum += (double)x[0] * (double)y[0] + (double)x[inc] * (double)y[inc];
                x += inc2;
                y += inc2;
                n -= 2;
            }

            return sum;
        }

        private static unsafe double CholDot(int n, double* x, double* y, int inc)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x += inc;
                    y += inc;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            var inc2 = 2 * inc;
            while (n > 0)
            {
                sum += x[0] * y[0] + x[inc] * y[inc];
                x += inc2;
                y += inc2;
                n -= 2;
            }

            return sum;
        }

        private static unsafe Complex CholDot(int n, Complex* x, Complex* y, int inc)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = Complex.Zero;
                    break;
                case 1:
                    sum = CholMul(x[0], y[0]);
                    x += inc;
                    y += inc;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            var inc2 = 2 * inc;
            while (n > 0)
            {
                sum += CholMul(x[0], y[0]) + CholMul(x[inc], y[inc]);
                x += inc2;
                y += inc2;
                n -= 2;
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Complex CholMul(Complex x, Complex y)
        {
            var a = x.Real;
            var b = -x.Imaginary;
            var c = y.Real;
            var d = y.Imaginary;
            return new Complex(a * c - b * d, a * d + b * c);
        }
    }
}
