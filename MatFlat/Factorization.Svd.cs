using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;

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
            var work = new Complex[rowsA];
            var e = new Complex[columnsA];
            var stmp = new Complex[Math.Min(rowsA + 1, columnsA)];

            fixed (Complex* pwork = work)
            fixed (Complex* pe = e)
            fixed (Complex* pstmp = stmp)
            {
                SvdCore(rowsA, columnsA, a, lda, s, u, ldu, vt, ldvt, pwork, pe, pstmp);
            }
        }

        public static unsafe void SvdCore(int rowsA, int columnsA, Complex* a, int lda, Complex* s, Complex* u, int ldu, Complex* vt, int ldvt, Complex* work, Complex* e, Complex* stmp)
        {
            var computeVectors = true;

            const int maxiter = 1000;

            for (var ccc = 0; ccc < columnsA; ccc++)
            {
                new Span<Complex>(vt + ccc * ldvt, columnsA).Clear();
            }

            int i2, j2, l;

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

                if (k < nct)
                {
                    // Compute the transformation for the l-th column and
                    // place the l-th diagonal in vector s[l].
                    stmp[k] = SvdNorm(rowsA - k, aColk + k);

                    if (stmp[k] != Complex.Zero)
                    {
                        if (aColk[k] != Complex.Zero)
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
                        if (stmp[k] != Complex.Zero)
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
                    // Place the transformation in "u" for subsequent back multiplication.
                    var uColk = u + ldu * k;
                    var copyLength = sizeof(Complex) * (rowsA - k);
                    Buffer.MemoryCopy(aColk + k, uColk + k, copyLength, copyLength);
                }

                if (k >= nrt)
                {
                    continue;
                }

                // Compute the l-th row transformation and place the l-th super-diagonal in e(l).
                e[k] = SvdNorm(columnsA - kp1, e + kp1);
                if (e[k] != Complex.Zero)
                {
                    if (e[kp1] != Complex.Zero)
                    {
                        e[k] = ChangeArgument(e[k], e[kp1]);
                    }

                    // Scale vector "e" from "lp1" by 1.0 / e[l]
                    SvdDivInplace(columnsA - kp1, e + kp1, e[k]);

                    e[kp1] += Complex.One;
                }
                e[k] = new Complex(-e[k].Real, e[k].Imaginary);

                if (kp1 < rowsA && e[k] != Complex.Zero)
                {
                    // Apply the transformation.
                    new Span<Complex>(work + kp1, rowsA - kp1).Clear();
                    for (var j = kp1; j < columnsA; j++)
                    {
                        var aColj = a + lda * j;
                        SvdMulAdd(rowsA - kp1, aColj + kp1, e[j], work + kp1);
                    }

                    for (var j = kp1; j < columnsA; j++)
                    {
                        var aColj = a + lda * j;
                        SvdMulAdd(rowsA - kp1, work + kp1, (-e[j] / e[kp1]).Conjugate(), aColj + kp1);
                    }
                }

                if (computeVectors)
                {
                    // Place the transformation in v for subsequent back multiplication.
                    var vtColk = vt + ldvt * k;
                    var copyLength = sizeof(Complex) * (columnsA - kp1);
                    Buffer.MemoryCopy(e + kp1, vtColk + kp1, copyLength, copyLength);
                }
            }

            // Set up the final bidiagonal matrix or order m.
            var p = Math.Min(columnsA, rowsA + 1);
            if (nct < columnsA)
            {
                stmp[nct] = a[(nct * lda) + nct];
            }
            if (rowsA < p)
            {
                stmp[p - 1] = Complex.Zero;
            }
            if (nrt + 1 < p)
            {
                e[nrt] = a[((p - 1) * lda) + nrt];
            }
            e[p - 1] = Complex.Zero;

            // If required, generate "u".
            if (computeVectors)
            {
                for (var j = nct; j < ncu; j++)
                {
                    var uColj = u + ldu * j;
                    new Span<Complex>(uColj, rowsA).Clear();
                    uColj[j] = Complex.One;
                }

                for (var k = nct - 1; k >= 0; k--)
                {
                    var uColk = u + ldu * k;

                    if (stmp[k] != Complex.Zero)
                    {
                        for (var j = k + 1; j < ncu; j++)
                        {
                            var uColj = u + ldu * j;
                            var t = -SvdDot(rowsA - k, uColk + k, uColj + k) / uColk[k];
                            SvdMulAdd(rowsA - k, uColk + k, t, uColj + k);
                        }

                        // A part of column "l" of matrix A from row "l" to end multiply by -1.0
                        SvdFlipSign(rowsA - k, uColk + k);

                        uColk[k] += Complex.One;
                        new Span<Complex>(uColk, k).Clear();
                    }
                    else
                    {
                        new Span<Complex>(uColk, rowsA).Clear();
                        uColk[k] = Complex.One;
                    }
                }
            }

            // If it is required, generate v.
            if (computeVectors)
            {
                for (var k = columnsA - 1; k >= 0; k--)
                {
                    var kp1 = k + 1;
                    var vtColl = vt + ldvt * k;

                    if (k < nrt)
                    {
                        if (e[k] != Complex.Zero)
                        {
                            for (var j = kp1; j < columnsA; j++)
                            {
                                var vtColj = vt + ldvt * j;
                                var t = -SvdDot(columnsA - kp1, vtColl + kp1, vtColj + kp1) / vtColl[kp1];
                                SvdMulAdd(columnsA - k, vtColl + k, t, vtColj + k);
                            }
                        }
                    }

                    new Span<Complex>(vtColl, columnsA).Clear();
                    vtColl[k] = Complex.One;
                }
            }

            // Transform "s" and "e" so that they are double
            for (var i = 0; i < p; i++)
            {
                if (stmp[i] != Complex.Zero)
                {
                    var t3 = stmp[i].Magnitude;
                    var r = stmp[i] / t3;
                    stmp[i] = t3;
                    if (i < p - 1)
                    {
                        e[i] /= r;
                    }

                    if (computeVectors)
                    {
                        // A part of column "i" of matrix U from row 0 to end multiply by r
                        SvdMulInplace(rowsA, u + ldu * i, r);
                    }
                }

                // Exit
                if (i == p - 1)
                {
                    break;
                }

                if (e[i] == 0.0)
                {
                    continue;
                }

                var t = e[i].Magnitude;
                var r2 = t / e[i];
                e[i] = t;
                stmp[i + 1] = stmp[i + 1] * r2;

                if (computeVectors)
                {
                    // A part of column "i+1" of matrix VT from row 0 to end multiply by r
                    SvdMulInplace(columnsA, vt + ldvt * (i + 1), r2);
                }
            }

            // Main iteration loop for the singular values.
            var mn = p;
            var iter = 0;

            while (p > 0)
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
                for (l = p - 2; l >= 0; l--)
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
                if (l == p - 2)
                {
                    kase = 4;
                }
                else
                {
                    int ls;
                    for (ls = p - 1; ls > l; ls--)
                    {
                        test = 0.0;
                        if (ls != p - 1)
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
                    else if (ls == p - 1)
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
                        f = e[p - 2].Real;
                        e[p - 2] = 0.0;
                        double t1;
                        for (var kk = l; kk < p - 1; kk++)
                        {
                            k = p - 2 - kk + l;
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
                                    var z = (cs * vt[(k * ldvt) + i2]) + (sn * vt[((p - 1) * ldvt) + i2]);
                                    vt[((p - 1) * ldvt) + i2] = (cs * vt[((p - 1) * ldvt) + i2]) - (sn * vt[(k * ldvt) + i2]);
                                    vt[(k * ldvt) + i2] = z;
                                }
                            }
                        }

                        break;

                    // Split at negligible s[l].
                    case 2:
                        f = e[l - 1].Real;
                        e[l - 1] = 0.0;
                        for (k = l; k < p; k++)
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
                        scale = Math.Max(scale, stmp[p - 1].Magnitude);
                        scale = Math.Max(scale, stmp[p - 2].Magnitude);
                        scale = Math.Max(scale, e[p - 2].Magnitude);
                        scale = Math.Max(scale, stmp[l].Magnitude);
                        scale = Math.Max(scale, e[l].Magnitude);
                        var sm = stmp[p - 1].Real / scale;
                        var smm1 = stmp[p - 2].Real / scale;
                        var emm1 = e[p - 2].Real / scale;
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
                        for (k = l; k < p - 1; k++)
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

                        e[p - 2] = f;
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
                        p = p - 1;
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
            new Span<Complex>(stmp, Math.Min(rowsA, columnsA)).CopyTo(new Span<Complex>(s, Math.Min(rowsA, columnsA)));
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

        private static unsafe void SvdMulInplace(int n, Complex* x, Complex y)
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
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] *= y;
                x[1] *= y;
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

        private static unsafe void SvdFlipSign<T>(int n, T* x) where T : unmanaged, INumberBase<T>
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
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                x[0] = -x[0];
                x[1] = -x[1];
                x += 2;
                n -= 2;
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
