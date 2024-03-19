using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace MatFlat
{
    public static partial class Factorization
    {
        public static Complex Conjugate(this Complex value)
        {
            return new Complex(value.Real, -value.Imaginary);
        }

        public static bool AlmostEqualRelative(this double a, double b, double maximumError)
        {
            return Math.Abs(a - b) < 1.0E-15;
        }

        public static unsafe void SvdComplex(int rowsA, int columnsA, Complex* a, int lda, Complex* s, Complex* u, int ldu, Complex* vt, int ldvt)
        {
            var computeVectors = true;

            var work = new Complex[rowsA];

            const int maxiter = 1000;

            var e = new Complex[columnsA];
            var stmp = new Complex[Math.Min(rowsA + 1, columnsA)];

            for (var ccc = 0; ccc < columnsA; ccc++)
            {
                new Span<Complex>(vt + ccc * ldvt, columnsA).Clear();
            }

            int i2, j2, l, lp1;

            Complex t2;

            var ncu = rowsA;

            // Reduce matrix to bidiagonal form, storing the diagonal elements
            // in "s" and the super-diagonal elements in "e".
            var nct = Math.Min(rowsA - 1, columnsA);
            var nrt = Math.Max(0, Math.Min(columnsA - 2, rowsA));
            var kmax = Math.Max(nct, nrt);
            for (var k = 0; k < kmax; k++)
            {
                var kp1 = k + 1;
                var aColk = a + lda * k;

                lp1 = k + 1;
                if (k < nct)
                {
                    // Compute the transformation for the l-th column and
                    // place the l-th diagonal in vector s[l].
                    stmp[k] = SvdNorm(rowsA - k, aColk + k);

                    if (stmp[k] != 0.0)
                    {
                        if (aColk[k] != 0.0)
                        {
                            stmp[k] = ChangeArgument(stmp[k], aColk[k]);
                        }

                        // A part of column "l" of Matrix A from row "l" to end multiply by 1.0 / s[l]
                        SvdDivInplace(rowsA - k, aColk + k, stmp[k]);
                        aColk[k] += 1.0;
                    }

                    stmp[k] = -stmp[k];
                }

                for (var j = kp1; j < columnsA; j++)
                {
                    var aColj = a + lda * j;

                    if (k < nct)
                    {
                        if (stmp[k] != 0.0)
                        {
                            // Apply the transformation.
                            var t = -SvdDot(rowsA - k, aColk + k, aColj + k) / aColk[k];
                            SvdMulAdd(rowsA - k, aColk + k, t, aColj + k);
                        }
                    }

                    // Place the l-th row of matrix into "e" for the
                    // subsequent calculation of the row transformation.
                    e[j] = aColj[k].Conjugate();
                }

                if (computeVectors && k < nct)
                {
                    var uColk = u + ldu * k;

                    // Place the transformation in "u" for subsequent back multiplication.
                    var copyLength = sizeof(Complex) * (rowsA - k);
                    Buffer.MemoryCopy(aColk + k, uColk + k, copyLength, copyLength);
                }

                if (k >= nrt)
                {
                    continue;
                }

                // Compute the l-th row transformation and place the l-th super-diagonal in e(l).
                e[k] = SvdNorm(e.AsSpan(kp1));
                if (e[k] != 0.0)
                {
                    if (e[kp1] != 0.0)
                    {
                        e[k] = ChangeArgument(e[k], e[kp1]);
                    }

                    // Scale vector "e" from "lp1" by 1.0 / e[l]
                    SvdDivInplace(e.AsSpan(kp1), e[k]);

                    e[kp1] += 1.0;
                }
                e[k] = new Complex(-e[k].Real, e[k].Imaginary);

                if (kp1 < rowsA && e[k] != 0.0)
                {
                    // Apply the transformation.
                    work.AsSpan(kp1).Clear();
                    for (var j = kp1; j < columnsA; j++)
                    {
                        var aColj = a + lda * j;
                        SvdMulAdd(aColj + kp1, e[j], work.AsSpan(kp1));
                    }

                    for (var j = kp1; j < columnsA; j++)
                    {
                        var aColj = a + lda * j;
                        SvdMulAdd(work.AsSpan(kp1), (-e[j] / e[kp1]).Conjugate(), aColj + kp1);
                    }
                }

                if (computeVectors)
                {
                    // Place the transformation in v for subsequent back multiplication.
                    for (i2 = lp1; i2 < columnsA; i2++)
                    {
                        vt[(k * ldvt) + i2] = e[i2];
                    }
                }
            }

            // Set up the final bidiagonal matrix or order m.
            var m = Math.Min(columnsA, rowsA + 1);
            var nctp1 = nct + 1;
            var nrtp1 = nrt + 1;
            if (nct < columnsA)
            {
                stmp[nctp1 - 1] = a[((nctp1 - 1) * lda) + (nctp1 - 1)];
            }

            if (rowsA < m)
            {
                stmp[m - 1] = 0.0;
            }

            if (nrtp1 < m)
            {
                e[nrtp1 - 1] = a[((m - 1) * lda) + (nrtp1 - 1)];
            }

            e[m - 1] = 0.0;

            // If required, generate "u".
            if (computeVectors)
            {
                for (j2 = nctp1 - 1; j2 < ncu; j2++)
                {
                    for (i2 = 0; i2 < rowsA; i2++)
                    {
                        u[(j2 * ldu) + i2] = 0.0;
                    }

                    u[(j2 * ldu) + j2] = 1.0;
                }

                for (l = nct - 1; l >= 0; l--)
                {
                    if (stmp[l] != 0.0)
                    {
                        for (j2 = l + 1; j2 < ncu; j2++)
                        {
                            t2 = 0.0;
                            for (i2 = l; i2 < rowsA; i2++)
                            {
                                t2 += u[(l * ldu) + i2].Conjugate() * u[(j2 * ldu) + i2];
                            }

                            t2 = -t2 / u[(l * ldu) + l];
                            for (var ii = l; ii < rowsA; ii++)
                            {
                                u[(j2 * ldu) + ii] += t2 * u[(l * ldu) + ii];
                            }
                        }

                        // A part of column "l" of matrix A from row "l" to end multiply by -1.0
                        for (i2 = l; i2 < rowsA; i2++)
                        {
                            u[(l * ldu) + i2] = u[(l * ldu) + i2] * -1.0;
                        }

                        u[(l * ldu) + l] = 1.0 + u[(l * ldu) + l];
                        for (i2 = 0; i2 < l; i2++)
                        {
                            u[(l * ldu) + i2] = 0.0;
                        }
                    }
                    else
                    {
                        for (i2 = 0; i2 < rowsA; i2++)
                        {
                            u[(l * ldu) + i2] = 0.0;
                        }

                        u[(l * ldu) + l] = 1.0;
                    }
                }
            }

            // If it is required, generate v.
            if (computeVectors)
            {
                for (l = columnsA - 1; l >= 0; l--)
                {
                    lp1 = l + 1;
                    if (l < nrt)
                    {
                        if (e[l] != 0.0)
                        {
                            for (j2 = lp1; j2 < columnsA; j2++)
                            {
                                t2 = 0.0;
                                for (i2 = lp1; i2 < columnsA; i2++)
                                {
                                    t2 += vt[(l * ldvt) + i2].Conjugate() * vt[(j2 * ldvt) + i2];
                                }

                                t2 = -t2 / vt[(l * ldvt) + lp1];
                                for (var ii = l; ii < columnsA; ii++)
                                {
                                    vt[(j2 * ldvt) + ii] += t2 * vt[(l * ldvt) + ii];
                                }
                            }
                        }
                    }

                    for (i2 = 0; i2 < columnsA; i2++)
                    {
                        vt[(l * ldvt) + i2] = 0.0;
                    }

                    vt[(l * ldvt) + l] = 1.0;
                }
            }

            // Transform "s" and "e" so that they are double
            for (i2 = 0; i2 < m; i2++)
            {
                Complex r;
                if (stmp[i2] != 0.0)
                {
                    t2 = stmp[i2].Magnitude;
                    r = stmp[i2] / t2;
                    stmp[i2] = t2;
                    if (i2 < m - 1)
                    {
                        e[i2] = e[i2] / r;
                    }

                    if (computeVectors)
                    {
                        // A part of column "i" of matrix U from row 0 to end multiply by r
                        for (j2 = 0; j2 < rowsA; j2++)
                        {
                            u[(i2 * ldu) + j2] = u[(i2 * ldu) + j2] * r;
                        }
                    }
                }

                // Exit
                if (i2 == m - 1)
                {
                    break;
                }

                if (e[i2] == 0.0)
                {
                    continue;
                }

                t2 = e[i2].Magnitude;
                r = t2 / e[i2];
                e[i2] = t2;
                stmp[i2 + 1] = stmp[i2 + 1] * r;
                if (!computeVectors)
                {
                    continue;
                }

                // A part of column "i+1" of matrix VT from row 0 to end multiply by r
                for (j2 = 0; j2 < columnsA; j2++)
                {
                    vt[((i2 + 1) * ldvt) + j2] = vt[((i2 + 1) * ldvt) + j2] * r;
                }
            }

            // Main iteration loop for the singular values.
            var mn = m;
            var iter = 0;

            while (m > 0)
            {
                // Quit if all the singular values have been found.
                // If too many iterations have been performed throw exception.
                if (iter >= maxiter)
                {
                    throw new Exception();
                }

                // This section of the program inspects for negligible elements in the s and e arrays,
                // on completion the variables case and l are set as follows:
                // case = 1: if mS[m] and e[l-1] are negligible and l < m
                // case = 2: if mS[l] is negligible and l < m
                // case = 3: if e[l-1] is negligible, l < m, and mS[l, ..., mS[m] are not negligible (qr step).
                // case = 4: if e[m-1] is negligible (convergence).
                double ztest;
                double test;
                for (l = m - 2; l >= 0; l--)
                {
                    test = stmp[l].Magnitude + stmp[l + 1].Magnitude;
                    ztest = test + e[l].Magnitude;
                    if (ztest.AlmostEqualRelative(test, 15))
                    {
                        e[l] = 0.0;
                        break;
                    }
                }

                int kase;
                if (l == m - 2)
                {
                    kase = 4;
                }
                else
                {
                    int ls;
                    for (ls = m - 1; ls > l; ls--)
                    {
                        test = 0.0;
                        if (ls != m - 1)
                        {
                            test = test + e[ls].Magnitude;
                        }

                        if (ls != l + 1)
                        {
                            test = test + e[ls - 1].Magnitude;
                        }

                        ztest = test + stmp[ls].Magnitude;
                        if (ztest.AlmostEqualRelative(test, 15))
                        {
                            stmp[ls] = 0.0;
                            break;
                        }
                    }

                    if (ls == l)
                    {
                        kase = 3;
                    }
                    else if (ls == m - 1)
                    {
                        kase = 1;
                    }
                    else
                    {
                        kase = 2;
                        l = ls;
                    }
                }

                l = l + 1;

                // Perform the task indicated by case.
                int k;
                double f;
                double cs;
                double sn;
                switch (kase)
                {
                    // Deflate negligible s[m].
                    case 1:
                        f = e[m - 2].Real;
                        e[m - 2] = 0.0;
                        double t1;
                        for (var kk = l; kk < m - 1; kk++)
                        {
                            k = m - 2 - kk + l;
                            t1 = stmp[k].Real;
                            Drotg(ref t1, ref f, out cs, out sn);
                            stmp[k] = t1;
                            if (k != l)
                            {
                                f = -sn * e[k - 1].Real;
                                e[k - 1] = cs * e[k - 1];
                            }

                            if (computeVectors)
                            {
                                // Rotate
                                for (i2 = 0; i2 < columnsA; i2++)
                                {
                                    var z = (cs * vt[(k * ldvt) + i2]) + (sn * vt[((m - 1) * ldvt) + i2]);
                                    vt[((m - 1) * ldvt) + i2] = (cs * vt[((m - 1) * ldvt) + i2]) - (sn * vt[(k * ldvt) + i2]);
                                    vt[(k * ldvt) + i2] = z;
                                }
                            }
                        }

                        break;

                    // Split at negligible s[l].
                    case 2:
                        f = e[l - 1].Real;
                        e[l - 1] = 0.0;
                        for (k = l; k < m; k++)
                        {
                            t1 = stmp[k].Real;
                            Drotg(ref t1, ref f, out cs, out sn);
                            stmp[k] = t1;
                            f = -sn * e[k].Real;
                            e[k] = cs * e[k];
                            if (computeVectors)
                            {
                                // Rotate
                                for (i2 = 0; i2 < rowsA; i2++)
                                {
                                    var z = (cs * u[(k * ldu) + i2]) + (sn * u[((l - 1) * ldu) + i2]);
                                    u[((l - 1) * ldu) + i2] = (cs * u[((l - 1) * ldu) + i2]) - (sn * u[(k * ldu) + i2]);
                                    u[(k * ldu) + i2] = z;
                                }
                            }
                        }

                        break;

                    // Perform one qr step.
                    case 3:
                        // calculate the shift.
                        var scale = 0.0;
                        scale = Math.Max(scale, stmp[m - 1].Magnitude);
                        scale = Math.Max(scale, stmp[m - 2].Magnitude);
                        scale = Math.Max(scale, e[m - 2].Magnitude);
                        scale = Math.Max(scale, stmp[l].Magnitude);
                        scale = Math.Max(scale, e[l].Magnitude);
                        var sm = stmp[m - 1].Real / scale;
                        var smm1 = stmp[m - 2].Real / scale;
                        var emm1 = e[m - 2].Real / scale;
                        var sl = stmp[l].Real / scale;
                        var el = e[l].Real / scale;
                        var b = (((smm1 + sm) * (smm1 - sm)) + (emm1 * emm1)) / 2.0;
                        var c = (sm * emm1) * (sm * emm1);
                        var shift = 0.0;
                        if (b != 0.0 || c != 0.0)
                        {
                            shift = Math.Sqrt((b * b) + c);
                            if (b < 0.0)
                            {
                                shift = -shift;
                            }

                            shift = c / (b + shift);
                        }

                        f = ((sl + sm) * (sl - sm)) + shift;
                        var g = sl * el;

                        // Chase zeros
                        for (k = l; k < m - 1; k++)
                        {
                            Drotg(ref f, ref g, out cs, out sn);
                            if (k != l)
                            {
                                e[k - 1] = f;
                            }

                            f = (cs * stmp[k].Real) + (sn * e[k].Real);
                            e[k] = (cs * e[k]) - (sn * stmp[k]);
                            g = sn * stmp[k + 1].Real;
                            stmp[k + 1] = cs * stmp[k + 1];
                            if (computeVectors)
                            {
                                for (i2 = 0; i2 < columnsA; i2++)
                                {
                                    var z = (cs * vt[(k * ldvt) + i2]) + (sn * vt[((k + 1) * ldvt) + i2]);
                                    vt[((k + 1) * ldvt) + i2] = (cs * vt[((k + 1) * ldvt) + i2]) - (sn * vt[(k * ldvt) + i2]);
                                    vt[(k * ldvt) + i2] = z;
                                }
                            }

                            Drotg(ref f, ref g, out cs, out sn);
                            stmp[k] = f;
                            f = (cs * e[k].Real) + (sn * stmp[k + 1].Real);
                            stmp[k + 1] = -(sn * e[k]) + (cs * stmp[k + 1]);
                            g = sn * e[k + 1].Real;
                            e[k + 1] = cs * e[k + 1];
                            if (computeVectors && k < rowsA)
                            {
                                for (i2 = 0; i2 < rowsA; i2++)
                                {
                                    var z = (cs * u[(k * ldu) + i2]) + (sn * u[((k + 1) * ldu) + i2]);
                                    u[((k + 1) * ldu) + i2] = (cs * u[((k + 1) * ldu) + i2]) - (sn * u[(k * ldu) + i2]);
                                    u[(k * ldu) + i2] = z;
                                }
                            }
                        }

                        e[m - 2] = f;
                        iter = iter + 1;
                        break;

                    // Convergence
                    case 4:

                        // Make the singular value  positive
                        if (stmp[l].Real < 0.0)
                        {
                            stmp[l] = -stmp[l];
                            if (computeVectors)
                            {
                                // A part of column "l" of matrix VT from row 0 to end multiply by -1
                                for (i2 = 0; i2 < columnsA; i2++)
                                {
                                    vt[(l * ldvt) + i2] = vt[(l * ldvt) + i2] * -1.0;
                                }
                            }
                        }

                        // Order the singular value.
                        while (l != mn - 1)
                        {
                            if (stmp[l].Real >= stmp[l + 1].Real)
                            {
                                break;
                            }

                            t2 = stmp[l];
                            stmp[l] = stmp[l + 1];
                            stmp[l + 1] = t2;
                            if (computeVectors && l < columnsA)
                            {
                                // Swap columns l, l + 1
                                for (i2 = 0; i2 < columnsA; i2++)
                                {
                                    (vt[(l * ldvt) + i2], vt[((l + 1) * ldvt) + i2]) = (vt[((l + 1) * ldvt) + i2], vt[(l * ldvt) + i2]);
                                }
                            }

                            if (computeVectors && l < rowsA)
                            {
                                // Swap columns l, l + 1
                                for (i2 = 0; i2 < rowsA; i2++)
                                {
                                    (u[(l * ldu) + i2], u[((l + 1) * ldu) + i2]) = (u[((l + 1) * ldu) + i2], u[(l * ldu) + i2]);
                                }
                            }
                            l = l + 1;
                        }
                        iter = 0;
                        m = m - 1;
                        break;
                }
            }

            if (computeVectors)
            {
                // Finally transpose "v" to get "vt" matrix
                for (i2 = 0; i2 < columnsA; i2++)
                {
                    for (j2 = 0; j2 <= i2; j2++)
                    {
                        if (j2 == i2)
                        {
                            vt[(j2 * ldvt) + i2] = vt[(j2 * ldvt) + i2].Conjugate();
                        }
                        else
                        {
                            var val1 = vt[(j2 * ldvt) + i2];
                            var val2 = vt[(i2 * ldvt) + j2];
                            vt[(j2 * ldvt) + i2] = val2.Conjugate();
                            vt[(i2 * ldvt) + j2] = val1.Conjugate();
                        }
                    }
                }
            }

            // Copy stemp to s with size adjustment. We are using ported copy of linpack's svd code and it uses
            // a singular vector of length rows+1 when rows < columns. The last element is not used and needs to be removed.
            // We should port lapack's svd routine to remove this problem.
            //Array.Copy(stemp, 0, s, 0, Math.Min(rowsA, columnsA));
            stmp.AsSpan(0, Math.Min(rowsA, columnsA)).CopyTo(new Span<Complex>(s, Math.Min(rowsA, columnsA)));
        }

        private static unsafe double SvdNorm(int n, Complex* x)
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
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0].Real * x[0].Real + x[0].Imaginary * x[0].Imaginary + x[1].Real * x[1].Real + x[1].Imaginary * x[1].Imaginary;
                x += 2;
                n -= 2;
            }

            return Math.Sqrt(sum);
        }

        private static unsafe double SvdNorm(ReadOnlySpan<Complex> x)
        {
            fixed (Complex* px = x)
            {
                return SvdNorm(x.Length, px);
            }
        }

        private static Complex ChangeArgument(Complex abs, Complex arg)
        {
            var num = abs.Real * abs.Real + abs.Imaginary * abs.Imaginary;
            var den = arg.Real * arg.Real + arg.Imaginary * arg.Imaginary;
            return Math.Sqrt(num / den) * arg;
        }

        private static unsafe void SvdDivInplace<T>(int n, T* x, T y) where T : unmanaged, INumberBase<T>
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

        private static unsafe void SvdDivInplace<T>(ReadOnlySpan<T> x, T y) where T : unmanaged, INumberBase<T>
        {
            fixed (T* px = x)
            {
                SvdDivInplace(x.Length, px, y);
            }
        }

        private static unsafe Complex SvdDot(int n, Complex* x, Complex* y)
        {
            Complex sum;
            switch (n & 1)
            {
                case 0:
                    sum = Complex.Zero;
                    break;
                case 1:
                    sum = SvdComplexMul(x[0], y[0]);
                    x++;
                    y++;
                    n--;
                    break;
                default:
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += SvdComplexMul(x[0], y[0]) + SvdComplexMul(x[1], y[1]);
                x += 2;
                y += 2;
                n -= 2;
            }

            return sum;
        }

        private static unsafe void SvdMulAdd<T>(int n, T* x, T y, T* dst) where T : unmanaged, INumberBase<T>
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

        private static unsafe void SvdMulAdd<T>(T* x, T y, Span<T> dst) where T : unmanaged, INumberBase<T>
        {
            fixed (T* pdst = dst)
            {
                SvdMulAdd(dst.Length, x, y, pdst);
            }
        }

        private static unsafe void SvdMulAdd<T>(Span<T> x, T y, T* dst) where T : unmanaged, INumberBase<T>
        {
            fixed (T* px = x)
            {
                SvdMulAdd(x.Length, px, y, dst);
            }
        }

        static void Drotg(ref double da, ref double db, out double c, out double s)
        {
            double r, z;

            var roe = db;
            var absda = Math.Abs(da);
            var absdb = Math.Abs(db);
            if (absda > absdb)
            {
                roe = da;
            }

            var scale = absda + absdb;
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
                if (absda > absdb)
                {
                    z = s;
                }

                if (absdb >= absda && c != 0.0)
                {
                    z = 1.0 / c;
                }
            }

            da = r;
            db = z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Complex SvdComplexMul(Complex x, Complex y)
        {
            var a = x.Real;
            var b = -x.Imaginary;
            var c = y.Real;
            var d = y.Imaginary;
            return new Complex(a * c - b * d, a * d + b * c);
        }
    }
}
