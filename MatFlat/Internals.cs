using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace MatFlat
{
    internal static class Internals
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex Conjugate(this Complex value)
        {
            return new Complex(value.Real, -value.Imaginary);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Complex MulConj(Complex x, Complex y)
        {
            var a = x.Real;
            var b = -x.Imaginary;
            var c = y.Real;
            var d = y.Imaginary;
            return new Complex(a * c - b * d, a * d + b * c);
        }

        internal static unsafe void DivInplace(int n, float* x, float y)
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

        internal static unsafe void DivInplace(int n, double* x, double y)
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

        internal static unsafe void DivInplace(int n, Complex* x, Complex y)
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

        internal static unsafe float Dot(int n, float* x, int incx, float* y)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0F;
                    break;
                case 1:
                    sum = (double)x[0] * (double)y[0];
                    x += incx;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += (double)x[0] * (double)y[0] + (double)x[incx] * (double)y[1];
                x += 2 * incx;
                y += 2;
                n -= 2;
            }

            return (float)sum;
        }

        internal static unsafe double Dot(int n, double* x, int incx, double* y)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x += incx;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * y[0] + x[incx] * y[1];
                x += 2 * incx;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe Complex Dot(int n, Complex* x, int incx, Complex* y)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x += incx;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * y[0] + x[incx] * y[1];
                x += 2 * incx;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe void SwapRows<T>(int n, T* x, T* y, int inc) where T : unmanaged, INumberBase<T>
        {
            while (n > 0)
            {
                (*x, *y) = (*y, *x);
                x += inc;
                y += inc;
                n--;
            }
        }

        internal static double FastMagnitude(Complex x)
        {
            return Math.Abs(x.Real) + Math.Abs(x.Imaginary);
        }
    }
}
