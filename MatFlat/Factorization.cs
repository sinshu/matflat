using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    /// <summary>
    /// Provides matrix factorization methods.
    /// </summary>
    public static partial class Factorization
    {
        private static double Hypotenuse(double a, double b)
        {
            var aa = Math.Abs(a);
            var ab = Math.Abs(b);

            if (aa > ab)
            {
                var r = b / a;
                return aa * Math.Sqrt(1.0 + r * r);
            }

            if (b != 0.0)
            {
                var r = a / b;
                return ab * Math.Sqrt(1.0 + r * r);
            }

            return 0.0;
        }
    }
}
