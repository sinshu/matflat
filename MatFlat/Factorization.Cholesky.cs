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

                var colk = a;

                for (var k = 0; k < j; k++)
                {
                    var t = colj[k] - Dot(k, a + j, a + k, lda);
                    t = t / colk[k];

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
                    throw new Exception("OMG");
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

        private static unsafe T Dot<T>(int n, T* x, T* y, int inc) where T : unmanaged, INumberBase<T>
        {
            T sum;
            switch (n & 1)
            {
                case 0:
                    sum = T.Zero;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x += inc;
                    y += inc;
                    n--;
                    break;
                default:
                    throw new Exception();
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
    }
}
