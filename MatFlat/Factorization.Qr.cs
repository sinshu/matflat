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

        public static unsafe void QrOrthogonalFactorDouble(int m, int n, double* qr, int ldqr, double* x, int ldx)
        {
            for (int k = n - 1; k >= 0; k--)
            {
                for (int i = 0; i < m; i++)
                {
                    x[i + ldx * k] = 0.0;
                }

                x[k + ldx * k] = 1.0;
                for (int j = k; j < n; j++)
                {
                    if (qr[k + ldqr * k] != 0)
                    {
                        double s = 0.0;

                        for (int i = k; i < m; i++)
                        {
                            s += qr[i + ldqr * k] * x[i + ldx * j];
                        }

                        s = -s / qr[k + ldqr * k];

                        for (int i = k; i < m; i++)
                        {
                            x[i + ldx * j] += s * qr[i + ldqr * k];
                        }
                    }
                }
            }
        }

        public static unsafe void QrUpperTriangularFactorDouble(int m, int n, double* qr, int ldqr, double* x, int ldx, double* rdiag)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i < j)
                    {
                        x[i + ldx * j] = qr[i + ldqr * j];
                    }
                    else if (i == j)
                    {
                        x[i + ldx * j] = rdiag[i];
                    }
                    else
                    {
                        x[i + ldx * j] = 0.0;
                    }
                }
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
    }
}
