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
        internal static double FastMagnitude(this Complex x)
        {
            return Math.Abs(x.Real) + Math.Abs(x.Imaginary);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex MulConj(Complex x, Complex y)
        {
            var a = x.Real;
            var b = -x.Imaginary;
            var c = y.Real;
            var d = y.Imaginary;
            return new Complex(a * c - b * d, a * d + b * c);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex ChangeArgument(Complex abs, Complex arg)
        {
            var num = abs.Real * abs.Real + abs.Imaginary * abs.Imaginary;
            var den = arg.Real * arg.Real + arg.Imaginary * arg.Imaginary;
            return Math.Sqrt(num / den) * arg;
        }

        internal static unsafe void MulAdd<T>(int n, T* x, T y, T* dst) where T : unmanaged, INumberBase<T>
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
                    throw new MatFlatException("An unexpected error occurred.");
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

        internal static unsafe void MulConjAdd(int n, Complex x, Complex* y, Complex* dst)
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    dst[0] += MulConj(x, y[0]);
                    y++;
                    dst++;
                    n--;
                    break;
                default:
                    throw new MatrixFactorizationException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                dst[0] += MulConj(x, y[0]);
                dst[1] += MulConj(x, y[1]);
                y += 2;
                dst += 2;
                n -= 2;
            }
        }

        internal static unsafe void MulInplace<T>(int n, T* x, T y) where T : unmanaged, INumberBase<T>
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    x[0] *= y;
                    x++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] *= y;
                x[1] *= y;
                x += 2;
                n -= 2;
            }
        }

        internal static unsafe void DivInplace<T>(int n, T* x, T y) where T : unmanaged, INumberBase<T>
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
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] /= y;
                x[1] /= y;
                x += 2;
                n -= 2;
            }
        }

        internal static unsafe double Dot(int n, float* x, float* y)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = (double)x[0] * (double)y[0];
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += (double)x[0] * (double)y[0] + (double)x[1] * (double)y[1];
                x += 2;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe double Dot(int n, double* x, double* y)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
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

        internal static unsafe Complex Dot(int n, Complex* x, Complex* y)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = Complex.Zero;
                    break;
                case 1:
                    sum = x[0] * y[0];
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
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

        internal static unsafe double Dot(int n, float* x, int incx, float* y, int incy)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = (double)x[0] * (double)y[0];
                    x += incx;
                    y += incy;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += (double)x[0] * (double)y[0] + (double)x[incx] * (double)y[incy];
                x += 2 * incx;
                y += 2 * incy;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe double Dot(int n, double* x, int incx, double* y, int incy)
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
                    y += incy;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * y[0] + x[incx] * y[incy];
                x += 2 * incx;
                y += 2 * incy;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe Complex Dot(int n, Complex* x, int incx, Complex* y, int incy)
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
                    y += incy;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * y[0] + x[incx] * y[incy];
                x += 2 * incx;
                y += 2 * incy;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe Complex DotConj(int n, Complex* x, Complex* y)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = Complex.Zero;
                    break;
                case 1:
                    sum = MulConj(x[0], y[0]);
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += MulConj(x[0], y[0]) + MulConj(x[1], y[1]);
                x += 2;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe Complex DotConj(int n, Complex* x, int incx, Complex* y, int incy)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = Complex.Zero;
                    break;
                case 1:
                    sum = MulConj(x[0], y[0]);
                    x += incx;
                    y += incy;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += MulConj(x[0], y[0]) + MulConj(x[incx], y[incy]);
                x += 2 * incx;
                y += 2 * incy;
                n -= 2;
            }

            return sum;
        }

        internal static unsafe double Norm(int n, float* x)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = (double)x[0] * (double)x[0];
                    x++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += (double)x[0] * (double)x[0] + (double)x[1] * (double)x[1];
                x += 2;
                n -= 2;
            }

            return Math.Sqrt(sum);
        }

        internal static unsafe double Norm(int n, double* x)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = x[0] * x[0];
                    x++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * x[0] + x[1] * x[1];
                x += 2;
                n -= 2;
            }

            return Math.Sqrt(sum);
        }

        internal static unsafe double Norm(int n, Complex* x)
        {
            double sum;
            switch (n & 1)
            {
                case 0:
                    sum = 0.0;
                    break;
                case 1:
                    sum = x[0].Real * x[0].Real + x[0].Imaginary * x[0].Imaginary;
                    x++;
                    n--;
                    break;
                default:
                    throw new MatFlatException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0].Real * x[0].Real + x[0].Imaginary * x[0].Imaginary + x[1].Real * x[1].Real + x[1].Imaginary * x[1].Imaginary;
                x += 2;
                n -= 2;
            }

            return Math.Sqrt(sum);
        }

        internal static unsafe void FlipSign<T>(int n, T* x) where T : unmanaged, INumberBase<T>
        {
            switch (n & 1)
            {
                case 0:
                    break;
                case 1:
                    x[0] = -x[0];
                    x++;
                    n--;
                    break;
                default:
                    throw new MatrixFactorizationException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] = -x[0];
                x[1] = -x[1];
                x += 2;
                n -= 2;
            }
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

        internal static void Srotg(ref float da, ref float db, out float c, out float s)
        {
            double r, z;

            var roe = (double)db;
            var ada = (double)Math.Abs(da);
            var adb = (double)Math.Abs(db);
            if (ada > adb)
            {
                roe = da;
            }

            var scale = ada + adb;
            if (scale == 0.0)
            {
                c = 1.0F;
                s = 0.0F;
                r = 0.0;
                z = 0.0;
            }
            else
            {
                var sda = da / scale;
                var sdb = db / scale;
                r = scale * Math.Sqrt((sda * sda) + (sdb * sdb));
                if (roe < 0.0F)
                {
                    r = -r;
                }

                c = (float)(da / r);
                s = (float)(db / r);
                z = 1.0F;
                if (ada > adb)
                {
                    z = s;
                }

                if (adb >= ada && c != 0.0F)
                {
                    z = 1.0F / c;
                }
            }

            da = (float)r;
            db = (float)z;
        }

        internal static void Drotg(ref double da, ref double db, out double c, out double s)
        {
            double r, z;

            var roe = db;
            var ada = Math.Abs(da);
            var adb = Math.Abs(db);
            if (ada > adb)
            {
                roe = da;
            }

            var scale = ada + adb;
            if (scale == 0.0)
            {
                c = 1.0;
                s = 0.0;
                r = 0.0;
                z = 0.0;
            }
            else
            {
                var sda = da / scale;
                var sdb = db / scale;
                r = scale * Math.Sqrt((sda * sda) + (sdb * sdb));
                if (roe < 0.0)
                {
                    r = -r;
                }

                c = da / r;
                s = db / r;
                z = 1.0;
                if (ada > adb)
                {
                    z = s;
                }

                if (adb >= ada && c != 0.0)
                {
                    z = 1.0 / c;
                }
            }

            da = r;
            db = z;
        }
    }
}
