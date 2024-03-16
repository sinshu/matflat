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

            for (var j = 0; j < n; j++)
            {
                var s = 0.0;
                for (var k = 0; k < j; k++)
                {
                    var t = a[k + j * lda];
                    for (var i = 0; i < k; i++)
                    {
                        t -= a[j + i * lda] * a[k + i * lda];
                    }
                    t = t / a[k + k * lda];

                    a[j + k * lda] = t;
                    s += t * t;
                }

                s = a[j + j * lda] - s;

                if (s > 0)
                {
                    a[j + j * lda] = Math.Sqrt(s);
                }
                else
                {
                    throw new Exception("OMG");
                }
            }

            for (var j = 1; j < n; j++)
            {
                new Span<double>(a + j * lda, j).Clear();
            }
        }
    }
}
