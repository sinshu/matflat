using System;
using System.Numerics;

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
        public static unsafe void Qr(int m, int n, float* a, int lda, float* rdiag)
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
                var norm = (float)Internals.Norm(m - k, colk + k);

                if (norm != 0.0F)
                {
                    Internals.DivInplace(m - k, colk + k, norm);

                    colk[k] += 1.0F;

                    // Apply transformation to remaining columns.
                    for (var j = k + 1; j < n; j++)
                    {
                        var colj = a + lda * j;
                        var s = -(float)(Internals.Dot(m - k, colk + k, colj + k) / colk[k]);
                        Internals.MulAdd(m - k, colk + k, s, colj + k);
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
        public static unsafe void Qr(int m, int n, double* a, int lda, double* rdiag)
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
                var norm = Internals.Norm(m - k, colk + k);

                if (norm != 0.0)
                {
                    Internals.DivInplace(m - k, colk + k, norm);

                    colk[k] += 1.0;

                    // Apply transformation to remaining columns.
                    for (var j = k + 1; j < n; j++)
                    {
                        var colj = a + lda * j;
                        var s = -Internals.Dot(m - k, colk + k, colj + k) / colk[k];
                        Internals.MulAdd(m - k, colk + k, s, colj + k);
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
        public static unsafe void Qr(int m, int n, Complex* a, int lda, double* rdiag)
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
                var norm = Internals.Norm(m - k, colk + k);

                if (norm != 0.0)
                {
                    Internals.DivInplace(m - k, colk + k, norm);

                    colk[k] += 1.0;

                    // Apply transformation to remaining columns.
                    for (var j = k + 1; j < n; j++)
                    {
                        var colj = a + lda * j;
                        var s = -Internals.DotConj(m - k, colk + k, colj + k) / colk[k];
                        Internals.MulAdd(m - k, colk + k, s, colj + k);
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
        /// The result of the QR decomposition obtained from <see cref="Qr(int, int, float*, int, float*)"/>.
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
        public static unsafe void QrOrthogonalFactor(int m, int n, float* a, int lda, float* q, int ldq)
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
                        var s = -(float)(Internals.Dot(m - k, aColk + k, qColj + k) / aColk[k]);
                        Internals.MulAdd(m - k, aColk + k, s, qColj + k);
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
        /// The result of the QR decomposition obtained from <see cref="Qr(int, int, double*, int, double*)"/>.
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
        public static unsafe void QrOrthogonalFactor(int m, int n, double* a, int lda, double* q, int ldq)
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
                        var s = -Internals.Dot(m - k, aColk + k, qColj + k) / aColk[k];
                        Internals.MulAdd(m - k, aColk + k, s, qColj + k);
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
        /// The result of the QR decomposition obtained from <see cref="Qr(int, int, Complex*, int, double*)"/>.
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
        public static unsafe void QrOrthogonalFactor(int m, int n, Complex* a, int lda, Complex* q, int ldq)
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
                        var s = -Internals.DotConj(m - k, qColj + k, aColk + k) / aColk[k];
                        Internals.MulConjAdd(m - k, s, aColk + k, qColj + k);
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
        /// The result of the QR decomposition obtained from <see cref="Qr(int, int, float*, int, float*)"/>.
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
        /// The diagonal elements of R obtained from <see cref="Qr(int, int, float*, int, float*)"/>.
        /// </param>
        public static unsafe void QrUpperTriangularFactor(int m, int n, float* a, int lda, float* r, int ldr, float* rdiag)
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
        /// The result of the QR decomposition obtained from <see cref="Qr(int, int, double*, int, double*)"/>.
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
        /// The diagonal elements of R obtained from <see cref="Qr(int, int, double*, int, double*)"/>.
        /// </param>
        public static unsafe void QrUpperTriangularFactor(int m, int n, double* a, int lda, double* r, int ldr, double* rdiag)
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
        /// The result of the QR decomposition obtained from <see cref="Qr(int, int, Complex*, int, double*)"/>.
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
        /// The diagonal elements of R obtained from <see cref="Qr(int, int, Complex*, int, double*)"/>.
        /// </param>
        public static unsafe void QrUpperTriangularFactor(int m, int n, Complex* a, int lda, Complex* r, int ldr, double* rdiag)
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
    }
}
