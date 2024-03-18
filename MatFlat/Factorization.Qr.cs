using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace MatFlat
{
    public static partial class Factorization
    {
        /// <summary>
        /// Computes a QR factorization of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the result of the QR decomposition are stored.
        /// The format of the decomposition result is identical to that of Mapack.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="rdiag">
        /// On exit, the diagonal elements of R are stored.
        /// </param>
        public static unsafe void QrSingle(int m, int n, float* a, int lda, float* rdiag)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (rdiag == null)
            {
                throw new ArgumentNullException(nameof(rdiag));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            var colk = a;

            for (var k = 0; k < n; k++)
            {
                var norm = QrNorm(m - k, colk + k);

                if (norm != 0.0)
                {
                    QrDivInplace(m - k, colk + k, norm);

                    colk[k] += 1.0F;

                    // Apply transformation to remaining columns.
                    for (var j = k + 1; j < n; j++)
                    {
                        var colj = a + lda * j;
                        var s = -QrDot(m - k, colk + k, colj + k) / colk[k];
                        QrMulAdd(m - k, colk + k, s, colj + k);
                    }
                }

                rdiag[k] = -norm;

                colk += lda;
            }
        }

        /// <summary>
        /// Computes a QR factorization of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the result of the QR decomposition are stored.
        /// The format of the decomposition result is identical to that of Mapack.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="rdiag">
        /// On exit, the diagonal elements of R are stored.
        /// </param>
        public static unsafe void QrDouble(int m, int n, double* a, int lda, double* rdiag)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (rdiag == null)
            {
                throw new ArgumentNullException(nameof(rdiag));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            var colk = a;

            for (var k = 0; k < n; k++)
            {
                var norm = QrNorm(m - k, colk + k);

                if (norm != 0.0)
                {
                    QrDivInplace(m - k, colk + k, norm);

                    colk[k] += 1.0;

                    // Apply transformation to remaining columns.
                    for (var j = k + 1; j < n; j++)
                    {
                        var colj = a + lda * j;
                        var s = -QrDot(m - k, colk + k, colj + k) / colk[k];
                        QrMulAdd(m - k, colk + k, s, colj + k);
                    }
                }

                rdiag[k] = -norm;

                colk += lda;
            }
        }

        /// <summary>
        /// Computes a QR factorization of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the result of the QR decomposition are stored.
        /// The format of the decomposition result is identical to that of Mapack.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="rdiag">
        /// On exit, the diagonal elements of R are stored.
        /// </param>
        public static unsafe void QrComplex(int m, int n, Complex* a, int lda, double* rdiag)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (rdiag == null)
            {
                throw new ArgumentNullException(nameof(rdiag));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            var colk = a;

            for (var k = 0; k < n; k++)
            {
                var norm = QrNorm(m - k, colk + k);

                if (norm != 0.0)
                {
                    QrDivInplace(m - k, colk + k, norm);

                    colk[k] += 1.0;

                    // Apply transformation to remaining columns.
                    for (var j = k + 1; j < n; j++)
                    {
                        var colj = a + lda * j;
                        var s = -QrDot(m - k, colk + k, colj + k) / colk[k];
                        QrMulAdd(m - k, colk + k, s, colj + k);
                    }
                }

                rdiag[k] = -norm;

                colk += lda;
            }
        }

        /// <summary>
        /// Gets the orthogonal factor Q from the QR decomposition result.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the source matrix.
        /// </param>
        /// <param name="n">
        /// The number of columns of the source matrix.
        /// </param>
        /// <param name="a">
        /// The result of the QR decomposition obtained from <see cref="QrSingle(int, int, float*, int, float*)"/>.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the source array.
        /// </param>
        /// <param name="q">
        /// On exit, the orthogonal factor Q is stored.
        /// </param>
        /// <param name="ldq">
        /// The leading dimension of the array Q.
        /// </param>
        public static unsafe void QrOrthogonalFactorSingle(int m, int n, float* a, int lda, float* q, int ldq)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (q == null)
            {
                throw new ArgumentNullException(nameof(q));
            }

            if (ldq < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldq));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            for (var k = n - 1; k >= 0; k--)
            {
                var qColk = q + ldq * k;
                var aColk = a + lda * k;

                new Span<float>(qColk, m).Clear();

                qColk[k] = 1.0F;

                for (var j = k; j < n; j++)
                {
                    var qColj = q + ldq * j;
                    if (aColk[k] != 0)
                    {
                        var s = -QrDot(m - k, aColk + k, qColj + k) / aColk[k];
                        QrMulAdd(m - k, aColk + k, s, qColj + k);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the orthogonal factor Q from the QR decomposition result.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the source matrix.
        /// </param>
        /// <param name="n">
        /// The number of columns of the source matrix.
        /// </param>
        /// <param name="a">
        /// The result of the QR decomposition obtained from <see cref="QrDouble(int, int, double*, int, double*)"/>.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the source array.
        /// </param>
        /// <param name="q">
        /// On exit, the orthogonal factor Q is stored.
        /// </param>
        /// <param name="ldq">
        /// The leading dimension of the array Q.
        /// </param>
        public static unsafe void QrOrthogonalFactorDouble(int m, int n, double* a, int lda, double* q, int ldq)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (q == null)
            {
                throw new ArgumentNullException(nameof(q));
            }

            if (ldq < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldq));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            for (var k = n - 1; k >= 0; k--)
            {
                var qColk = q + ldq * k;
                var aColk = a + lda * k;

                new Span<double>(qColk, m).Clear();

                qColk[k] = 1.0;

                for (var j = k; j < n; j++)
                {
                    var qColj = q + ldq * j;
                    if (aColk[k] != 0)
                    {
                        var s = -QrDot(m - k, aColk + k, qColj + k) / aColk[k];
                        QrMulAdd(m - k, aColk + k, s, qColj + k);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the orthogonal factor Q from the QR decomposition result.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the source matrix.
        /// </param>
        /// <param name="n">
        /// The number of columns of the source matrix.
        /// </param>
        /// <param name="a">
        /// The result of the QR decomposition obtained from <see cref="QrComplex(int, int, Complex*, int, double*)"/>.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the source array.
        /// </param>
        /// <param name="q">
        /// On exit, the orthogonal factor Q is stored.
        /// </param>
        /// <param name="ldq">
        /// The leading dimension of the array Q.
        /// </param>
        public static unsafe void QrOrthogonalFactorComplex(int m, int n, Complex* a, int lda, Complex* q, int ldq)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (q == null)
            {
                throw new ArgumentNullException(nameof(q));
            }

            if (ldq < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldq));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            for (var k = n - 1; k >= 0; k--)
            {
                var qColk = q + ldq * k;
                var aColk = a + lda * k;

                new Span<Complex>(qColk, m).Clear();

                qColk[k] = 1.0;

                for (var j = k; j < n; j++)
                {
                    var qColj = q + ldq * j;
                    if (aColk[k] != 0)
                    {
                        var s = -QrDot2(m - k, aColk + k, qColj + k) / aColk[k];
                        QrMulAdd2(m - k, aColk + k, s, qColj + k);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the upper triangular factor R from the QR decomposition result.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the source matrix.
        /// </param>
        /// <param name="n">
        /// The number of columns of the source matrix.
        /// </param>
        /// <param name="a">
        /// The result of the QR decomposition obtained from <see cref="QrSingle(int, int, float*, int, float*)"/>.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the source array.
        /// </param>
        /// <param name="r">
        /// On exit, the upper triangular factor R is stored.
        /// </param>
        /// <param name="ldr">
        /// The leading dimension of the array R.
        /// </param>
        /// <param name="rdiag">
        /// The diagonal elements of R obtained from <see cref="QrSingle(int, int, float*, int, float*)"/>.
        /// </param>
        public static unsafe void QrUpperTriangularFactorSingle(int m, int n, float* a, int lda, float* r, int ldr, float* rdiag)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (r == null)
            {
                throw new ArgumentNullException(nameof(r));
            }

            if (ldr < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldr));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            var aColi = a;
            var rColi = r;

            for (var i = 0; i < n; i++)
            {
                var copyLength = sizeof(float) * i;
                Buffer.MemoryCopy(aColi, rColi, copyLength, copyLength);

                rColi[i] = rdiag[i];

                var clearLength = sizeof(float) * (n - i - 1);
                new Span<float>(rColi + i + 1, n - i - 1).Clear();

                aColi += lda;
                rColi += ldr;
            }
        }

        /// <summary>
        /// Gets the upper triangular factor R from the QR decomposition result.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the source matrix.
        /// </param>
        /// <param name="n">
        /// The number of columns of the source matrix.
        /// </param>
        /// <param name="a">
        /// The result of the QR decomposition obtained from <see cref="QrDouble(int, int, double*, int, double*)"/>.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the source array.
        /// </param>
        /// <param name="r">
        /// On exit, the upper triangular factor R is stored.
        /// </param>
        /// <param name="ldr">
        /// The leading dimension of the array R.
        /// </param>
        /// <param name="rdiag">
        /// The diagonal elements of R obtained from <see cref="QrDouble(int, int, double*, int, double*)"/>.
        /// </param>
        public static unsafe void QrUpperTriangularFactorDouble(int m, int n, double* a, int lda, double* r, int ldr, double* rdiag)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (r == null)
            {
                throw new ArgumentNullException(nameof(r));
            }

            if (ldr < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldr));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            var aColi = a;
            var rColi = r;

            for (var i = 0; i < n; i++)
            {
                var copyLength = sizeof(double) * i;
                Buffer.MemoryCopy(aColi, rColi, copyLength, copyLength);

                rColi[i] = rdiag[i];

                var clearLength = sizeof(double) * (n - i - 1);
                new Span<double>(rColi + i + 1, n - i - 1).Clear();

                aColi += lda;
                rColi += ldr;
            }
        }


        /// <summary>
        /// Gets the upper triangular factor R from the QR decomposition result.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the source matrix.
        /// </param>
        /// <param name="n">
        /// The number of columns of the source matrix.
        /// </param>
        /// <param name="a">
        /// The result of the QR decomposition obtained from <see cref="QrComplex(int, int, Complex*, int, double*)"/>.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the source array.
        /// </param>
        /// <param name="r">
        /// On exit, the upper triangular factor R is stored.
        /// </param>
        /// <param name="ldr">
        /// The leading dimension of the array R.
        /// </param>
        /// <param name="rdiag">
        /// The diagonal elements of R obtained from <see cref="QrComplex(int, int, Complex*, int, double*)"/>.
        /// </param>
        public static unsafe void QrUpperTriangularFactorComplex(int m, int n, Complex* a, int lda, Complex* r, int ldr, double* rdiag)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (r == null)
            {
                throw new ArgumentNullException(nameof(r));
            }

            if (ldr < n)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldr));
            }

            if (n > m)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to the number of columns.");
            }

            var aColi = a;
            var rColi = r;

            for (var i = 0; i < n; i++)
            {
                var copyLength = sizeof(Complex) * i;
                Buffer.MemoryCopy(aColi, rColi, copyLength, copyLength);

                rColi[i] = rdiag[i];

                var clearLength = sizeof(Complex) * (n - i - 1);
                new Span<Complex>(rColi + i + 1, n - i - 1).Clear();

                aColi += lda;
                rColi += ldr;
            }
        }

        private static unsafe float QrNorm(int n, float* x)
        {
            float sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0F;
                    break;
                case 1:
                    sum = x[0] * x[0];
                    x++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * x[0] + x[1] * x[1];
                x += 2;
                n -= 2;
            }

            return MathF.Sqrt(sum);
        }

        private static unsafe double QrNorm(int n, double* x)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = x[0] * x[0];
                    x++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * x[0] + x[1] * x[1];
                x += 2;
                n -= 2;
            }

            return Math.Sqrt(sum);
        }

        private static unsafe double QrNorm(int n, Complex* x)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = x[0].Real * x[0].Real + x[0].Imaginary * x[0].Imaginary;
                    x++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0].Real * x[0].Real + x[0].Imaginary * x[0].Imaginary + x[1].Real * x[1].Real + x[1].Imaginary * x[1].Imaginary;
                x += 2;
                n -= 2;
            }

            return Math.Sqrt(sum);
        }

        private static unsafe void QrDivInplace<T>(int n, T* x, T y) where T : unmanaged, INumberBase<T>
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    x[0] /= y;
                    x++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] /= y;
                x[1] /= y;
                x += 2;
                n -= 2;
            }
        }

        private static unsafe void QrDivInplace(int n, Complex* x, double y)
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    x[0] /= y;
                    x++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] /= y;
                x[1] /= y;
                x += 2;
                n -= 2;
            }
        }

        private static unsafe T QrDot<T>(int n, T* x, T* y) where T : unmanaged, INumberBase<T>
        {
            T sum;
            switch (n & 1)
            {
                case 0:
                    sum = T.Zero;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * y[0] + x[1] * y[1];
                x += 2;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        private static unsafe void QrMulAdd<T>(int n, T* x, T y, T* dst) where T : unmanaged, INumberBase<T>
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    dst[0] += x[0] * y;
                    x++;
                    dst++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                dst[0] += x[0] * y;
                dst[1] += x[1] * y;
                x += 2;
                dst += 2;
                n -= 2;
            }
        }

        private static unsafe Complex QrDot(int n, Complex* x, Complex* y)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = Complex.Zero;
                    break;
                case 1:
                    sum = QrComplexMul(x[0], y[0]);
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += QrComplexMul(x[0], y[0]) + QrComplexMul(x[1], y[1]);
                x += 2;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        private static unsafe Complex QrDot2(int n, Complex* x, Complex* y)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = Complex.Zero;
                    break;
                case 1:
                    sum = QrComplexMul(y[0], x[0]);
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += QrComplexMul(y[0], x[0]) + QrComplexMul(y[1], x[1]);
                x += 2;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        private static unsafe void QrMulAdd2(int n, Complex* x, Complex y, Complex* dst)
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    dst[0] += QrComplexMul(y, x[0]);
                    x++;
                    dst++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                dst[0] += QrComplexMul(y, x[0]);
                dst[1] += QrComplexMul(y, x[1]);
                x += 2;
                dst += 2;
                n -= 2;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Complex QrComplexMul(Complex x, Complex y)
        {
            var a = x.Real;
            var b = -x.Imaginary;
            var c = y.Real;
            var d = y.Imaginary;
            return new Complex(a * c - b * d, a * d + b * c);
        }
    }
}
