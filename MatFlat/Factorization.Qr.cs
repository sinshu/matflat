using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        public static unsafe void QrDouble(int m, int n, double* a, int lda, double* rdiag)
        {
            var colk = a;

            for (var k = 0; k < n; k++)
            {
                // Compute 2-norm of k-th column without under/overflow.
                var norm = 0.0;
                for (var i = k; i < m; i++)
                {
                    norm = Hypotenuse(norm, colk[i]);
                }

                if (norm != 0.0)
                {
                    // Form k-th Householder vector.
                    if (colk[k] < 0)
                    {
                        norm = -norm;
                    }

                    for (var i = k; i < m; i++)
                    {
                        colk[i] /= norm;
                    }

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

        public static unsafe void QrComplex(int m, int n, Complex* a, int lda, double* rdiag)
        {
            var colk = a;

            for (var k = 0; k < n; k++)
            {
                // Compute 2-norm of k-th column without under/overflow.
                var norm = 0.0;
                for (var i = k; i < m; i++)
                {
                    norm = Hypotenuse(norm, colk[i]);
                }

                if (norm != 0.0)
                {
                    for (var i = k; i < m; i++)
                    {
                        colk[i] /= norm;
                    }

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

        public static unsafe void QrOrthogonalFactorDouble(int m, int n, double* a, int lda, double* q, int ldq)
        {
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

        public static unsafe void QrOrthogonalFactorComplex(int m, int n, Complex* a, int lda, Complex* q, int ldq)
        {
            for (var k = 0; k < n; k++)
            {
                var aColk = a + lda * k;
                for (var j = 0; j < m; j++)
                {
                    //aColk[j] = new Complex(aColk[j].Real, -aColk[j].Imaginary);
                }
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

            for (var k = 0; k < n; k++)
            {
                var qColk = q + ldq * k;
                for (var j = 0; j < m; j++)
                {
                    //qColk[j] = new Complex(qColk[j].Real, -qColk[j].Imaginary);
                }
            }
        }

        public static unsafe void QrUpperTriangularFactorDouble(int m, int n, double* a, int lda, double* r, int ldr, double* rdiag)
        {
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

        public static unsafe void QrUpperTriangularFactorComplex(int m, int n, Complex* a, int lda, Complex* r, int ldr, double* rdiag)
        {
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
            var sum = Complex.Zero;
            for (var i = 0; i < n; i++)
            {
                var cx = new Complex(x[i].Real, -x[i].Imaginary);
                var cy = new Complex(y[i].Real, y[i].Imaginary);
                sum += cx * cy;
            }
            return sum;
        }

        private static unsafe Complex QrDot2(int n, Complex* x, Complex* y)
        {
            var sum = Complex.Zero;
            for (var i = 0; i < n; i++)
            {
                var cx = new Complex(x[i].Real, x[i].Imaginary);
                var cy = new Complex(y[i].Real, -y[i].Imaginary);
                sum += cx * cy;
            }
            return sum;
        }

        private static unsafe void QrMulAdd(int n, Complex* x, Complex y, Complex* dst)
        {
            for (var i = 0; i < n; i++)
            {
                var cx = new Complex(x[i].Real, x[i].Imaginary);
                var cy = new Complex(y.Real, y.Imaginary);
                dst[i] += cx * cy;
            }
        }

        private static unsafe void QrMulAdd2(int n, Complex* x, Complex y, Complex* dst)
        {
            for (var i = 0; i < n; i++)
            {
                var cx = new Complex(x[i].Real, x[i].Imaginary);
                var cy = new Complex(y.Real, -y.Imaginary);
                dst[i] += cx * cy;
            }
        }
    }
}
