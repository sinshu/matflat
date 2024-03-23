using System;
using System.Numerics;

namespace MatFlat
{
    /// <summary>
    /// Provides a subset of the BLAS routines.
    /// </summary>
    public static partial class Blas
    {
        internal static unsafe void MulSub<T>(int n, T* x, int incx, T y, T* dst, int incdst) where T : unmanaged, INumberBase<T>
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    dst[0] -= x[0] * y;
                    x += incx;
                    dst += incdst;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                dst[0] -= x[0] * y;
                dst[incdst] -= x[incx] * y;
                x += 2 * incx;
                dst += 2 * incdst;
                n -= 2;
            }
        }
    }
}
