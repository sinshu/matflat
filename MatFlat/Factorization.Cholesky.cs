using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        public static unsafe void CholeskyDouble(int n, double* a, int lda)
        {
            var colj = a;

            for (int j = 0; j < n; j++)
            {
                Double s = 0;
                for (int k = 0; k < j; k++)
                {
                    Double t = a[k + j * lda];
                    for (int i = 0; i < k; i++)
                    {
                        t -= a[j + i * lda] * a[k + i * lda];
                    }
                    t = t / a[k + k * lda];

                    a[j + k * lda] = t;
                    s += t * t;
                }

                s = a[j + j * lda] - s;

                a[j + j * lda] = (Double)Math.Sqrt((double)s);
            }

            for (int j = 1; j < n; j++)
            {
                new Span<double>(a + j * lda, j).Clear();
            }
        }
    }
}
