using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        public static unsafe void QrDouble(int m, int n, double* a, int lda, double* rdiag)
        {
            for (int k = 0; k < n; k++)
            {
                // Compute 2-norm of k-th column without under/overflow.
                double nrm = 0;
                for (int i = k; i < m; i++)
                {
                    nrm = Hypotenuse(nrm, a[i + lda * k]);
                }

                if (nrm != 0.0)
                {
                    // Form k-th Householder vector.
                    if (a[k + lda * k] < 0)
                    {
                        nrm = -nrm;
                    }

                    for (int i = k; i < m; i++)
                    {
                        a[i + lda * k] /= nrm;
                    }

                    a[k + lda * k] += 1.0;

                    // Apply transformation to remaining columns.
                    for (int j = k + 1; j < n; j++)
                    {
                        double s = 0.0;

                        for (int i = k; i < m; i++)
                        {
                            s += a[i + lda * k] * a[i + lda * j];
                        }

                        s = -s / a[k + lda * k];

                        for (int i = k; i < m; i++)
                        {
                            a[i + lda * j] += s * a[i + lda * k];
                        }
                    }
                }

                rdiag[k] = -nrm;
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

        private static double Hypotenuse(double a, double b)
        {
            if (Math.Abs(a) > Math.Abs(b))
            {
                double r = b / a;
                return Math.Abs(a) * Math.Sqrt(1 + r * r);
            }

            if (b != 0)
            {
                double r = a / b;
                return Math.Abs(b) * Math.Sqrt(1 + r * r);
            }

            return 0.0;
        }
    }
}
