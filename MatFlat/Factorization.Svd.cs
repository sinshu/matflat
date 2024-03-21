using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace MatFlat
{
    public static partial class Factorization
    {
        public static bool AlmostEqualRelative(this double a, double b, double maximumError)
        {
            return Math.Abs(a - b) < 1.0E-15;
        }

        public static unsafe void Svd(int rowsA, int columnsA, double* a, int lda, double* s, double* u, int ldu, double* vt, int ldvt)
        {
            var work = new double[rowsA];
            var e = new double[columnsA];
            var stmp = new double[Math.Min(rowsA + 1, columnsA)];

            fixed (double* pwork = work)
            fixed (double* pe = e)
            fixed (double* pstmp = stmp)
            {
                SvdCore(rowsA, columnsA, a, lda, s, u, ldu, vt, ldvt, pwork, pe, pstmp);
            }
        }

        private static unsafe void SvdCore(int m, int n, double* a, int lda, double* s, double* u, int ldu, double* vt, int ldvt, double* work, double* e, double* stmp)
        {
            var computeVectors = true;

            int i2, j2;

            double t2;

            var ncu = m;

            // Reduce A to bidiagonal form, storing the diagonal elements in s and the super-diagonal elements in e.
            var nct = Math.Min(m - 1, n);
            var nrt = Math.Max(0, Math.Min(n - 2, m));
            var kmax = Math.Max(nct, nrt);
            for (var k = 0; k < kmax; k++)
            {
                var kp1 = k + 1;
                var aColk = a + lda * k;

                if (k < nct)
                {
                    // Compute the transformation for the k-th column and place the k-th diagonal in s[k].
                    // Compute 2-norm of k-th column.
                    stmp[k] = SvdNorm(m - k, aColk + k);

                    if (stmp[k] != 0.0)
                    {
                        if (aColk[k] != 0.0)
                        {
                            stmp[k] = Math.Abs(stmp[k]) * (aColk[k] / Math.Abs(aColk[k]));
                        }

                        SvdDivInplace(m - k, aColk + k, stmp[k]);

                        aColk[k] += 1.0;
                    }

                    stmp[k] = -stmp[k];
                }

                for (var j = kp1; j < n; j++)
                {
                    var aColj = a + lda * j;

                    if (k < nct && stmp[k] != 0.0)
                    {
                        // Apply the transformation.
                        var t = -SvdDot(m - k, aColk + k, aColj + k) / aColk[k];
                        SvdMulAdd(m - k, aColk + k, t, aColj + k);
                    }

                    // Place the k-th row of A into e for the subsequent calculation of the row transformation.
                    e[j] = aColj[k];
                }

                if (u != null && k < nct)
                {
                    // Place the transformation in U for subsequent back multiplication.
                    var uColk = u + ldu * k;
                    var copyLength = sizeof(double) * (m - k);
                    Buffer.MemoryCopy(aColk + k, uColk + k, copyLength, copyLength);
                }

                if (k < nrt)
                {
                    // Compute the k-th row transformation and place the k-th super-diagonal in e[k].
                    // Compute 2-norm.
                    e[k] = SvdNorm(n - kp1, e + kp1);

                    if (e[k] != 0.0)
                    {
                        if (e[kp1] != 0.0)
                        {
                            e[k] = Math.Abs(e[k]) * (e[kp1] / Math.Abs(e[kp1]));
                        }

                        SvdDivInplace(n - kp1, e + kp1, e[k]);

                        e[kp1] += 1.0;
                    }

                    e[k] = -e[k];

                    if (kp1 < m && e[k] != 0.0)
                    {
                        // Apply the transformation.
                        new Span<double>(work + kp1, m - kp1).Clear();

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            SvdMulAdd(m - kp1, aColj + kp1, e[j], work + kp1);
                        }

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            SvdMulAdd(m - kp1, work + kp1, -e[j] / e[kp1], aColj + kp1);
                        }
                    }

                    if (vt != null)
                    {
                        // Place the transformation in V for subsequent back multiplication.
                        var vtColk = vt + ldvt * k;
                        vtColk[k] = 0.0;
                        var copyLength = sizeof(double) * (n - kp1);
                        Buffer.MemoryCopy(e + kp1, vtColk + kp1, copyLength, copyLength);
                    }
                }
            }

            // Set up the final bidiagonal matrix or order p.
            var p = Math.Min(n, m + 1);
            if (nct < n)
            {
                stmp[nct] = a[(nct * lda) + nct];
            }
            if (m < p)
            {
                stmp[p - 1] = 0.0;
            }
            if (nrt + 1 < p)
            {
                e[nrt] = a[((p - 1) * lda) + nrt];
            }

            e[p - 1] = 0.0;

            // If required, generate U.
            if (u != null)
            {
                for (var j = nct; j < m; j++)
                {
                    var uColj = u + ldu * j;
                    new Span<double>(uColj, m).Clear();
                    uColj[j] = 1.0;
                }

                for (var k = nct - 1; k >= 0; k--)
                {
                    var uColk = u + ldu * k;

                    if (stmp[k] != 0.0)
                    {
                        for (var j = k + 1; j < m; j++)
                        {
                            var uColj = u + ldu * j;
                            var t = -SvdDot(m - k, uColk + k, uColj + k) / uColk[k];
                            SvdMulAdd(m - k, uColk + k, t, uColj + k);
                        }

                        SvdFlipSign(m - k, uColk + k);

                        uColk[k] += 1.0;
                        new Span<double>(uColk, k).Clear();
                    }
                    else
                    {
                        new Span<double>(uColk, m).Clear();
                        uColk[k] = 1.0;
                    }
                }
            }

            // If required, generate V.
            if (vt != null)
            {
                for (var k = n - 1; k >= 0; k--)
                {
                    var kp1 = k + 1;
                    var vtColk = vt + ldvt * k;

                    if (k < nrt && e[k] != 0.0)
                    {
                        for (var j = kp1; j < n; j++)
                        {
                            var vtColj = vt + ldvt * j;
                            var t = -SvdDot(n - kp1, vtColk + kp1, vtColj + kp1) / vtColk[kp1];
                            SvdMulAdd(n - k, vtColk + k, t, vtColj + k);
                        }
                    }

                    new Span<double>(vtColk, n).Clear();
                    vtColk[k] = 1.0;
                }
            }

            for (var i = 0; i < p; i++)
            {
                if (stmp[i] != 0.0)
                {
                    var t = stmp[i];
                    var r = stmp[i] / t;
                    stmp[i] = t;
                    if (i < p - 1)
                    {
                        e[i] /= r;
                    }

                    if (u != null)
                    {
                        SvdMulInplace(m, u + ldu * i, r);
                    }
                }

                if (i == p - 1)
                {
                    break;
                }

                if (e[i] != 0.0)
                {
                    var t = e[i];
                    var r = t / e[i];
                    e[i] = t;
                    stmp[i + 1] = stmp[i + 1] * r;

                    if (vt != null)
                    {
                        SvdMulInplace(n, vt + ldvt * (i + 1), r);
                    }
                }
            }

            // Main iteration loop for the singular values.
            var mn = p;
            var iter = 0;

            while (p > 0)
            {
                // Quit if all the singular values have been found.
                // If too many iterations have been performed throw exception.
                if (iter >= 1000)
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
                int k;
                for (k = p - 2; k >= 0; k--)
                {
                    test = Math.Abs(stmp[k]) + Math.Abs(stmp[k + 1]);
                    ztest = test + Math.Abs(e[k]);
                    if (ztest.AlmostEqualRelative(test, 15))
                    {
                        e[k] = 0.0;
                        break;
                    }
                }

                int kase;
                if (k == p - 2)
                {
                    kase = 4;
                }
                else
                {
                    int ls;
                    for (ls = p - 1; ls > k; ls--)
                    {
                        test = 0.0;
                        if (ls != p - 1)
                        {
                            test = test + Math.Abs(e[ls]);
                        }

                        if (ls != k + 1)
                        {
                            test = test + Math.Abs(e[ls - 1]);
                        }

                        ztest = test + Math.Abs(stmp[ls]);
                        if (ztest.AlmostEqualRelative(test, 15))
                        {
                            stmp[ls] = 0.0;
                            break;
                        }
                    }

                    if (ls == k)
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
                        k = ls;
                    }
                }

                k = k + 1;

                // Perform the task indicated by case.
                int k2;
                double f;
                double cs;
                double sn;
                switch (kase)
                {
                    // Deflate negligible s[m].
                    case 1:
                        f = e[p - 2];
                        e[p - 2] = 0.0;
                        double t1;
                        for (var kk = k; kk < p - 1; kk++)
                        {
                            k2 = p - 2 - kk + k;
                            t1 = stmp[k2];

                            Drotg(ref t1, ref f, out cs, out sn);
                            stmp[k2] = t1;
                            if (k2 != k)
                            {
                                f = -sn * e[k2 - 1];
                                e[k2 - 1] = cs * e[k2 - 1];
                            }

                            if (computeVectors)
                            {
                                // Rotate
                                for (i2 = 0; i2 < n; i2++)
                                {
                                    var z = (cs * vt[(k2 * ldvt) + i2]) + (sn * vt[((p - 1) * ldvt) + i2]);
                                    vt[((p - 1) * ldvt) + i2] = (cs * vt[((p - 1) * ldvt) + i2]) - (sn * vt[(k2 * ldvt) + i2]);
                                    vt[(k2 * ldvt) + i2] = z;
                                }
                            }
                        }

                        break;

                    // Split at negligible s[l].
                    case 2:
                        f = e[k - 1];
                        e[k - 1] = 0.0;
                        for (k2 = k; k2 < p; k2++)
                        {
                            t1 = stmp[k2];
                            Drotg(ref t1, ref f, out cs, out sn);
                            stmp[k2] = t1;
                            f = -sn * e[k2];
                            e[k2] = cs * e[k2];
                            if (computeVectors)
                            {
                                // Rotate
                                for (i2 = 0; i2 < m; i2++)
                                {
                                    var z = (cs * u[(k2 * ldu) + i2]) + (sn * u[((k - 1) * ldu) + i2]);
                                    u[((k - 1) * ldu) + i2] = (cs * u[((k - 1) * ldu) + i2]) - (sn * u[(k2 * ldu) + i2]);
                                    u[(k2 * ldu) + i2] = z;
                                }
                            }
                        }

                        break;

                    // Perform one qr step.
                    case 3:

                        // calculate the shift.
                        var scale = 0.0;
                        scale = Math.Max(scale, Math.Abs(stmp[p - 1]));
                        scale = Math.Max(scale, Math.Abs(stmp[p - 2]));
                        scale = Math.Max(scale, Math.Abs(e[p - 2]));
                        scale = Math.Max(scale, Math.Abs(stmp[k]));
                        scale = Math.Max(scale, Math.Abs(e[k]));
                        var sm = stmp[p - 1] / scale;
                        var smm1 = stmp[p - 2] / scale;
                        var emm1 = e[p - 2] / scale;
                        var sl = stmp[k] / scale;
                        var el = e[k] / scale;
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
                        for (k2 = k; k2 < p - 1; k2++)
                        {
                            Drotg(ref f, ref g, out cs, out sn);
                            if (k2 != k)
                            {
                                e[k2 - 1] = f;
                            }

                            f = (cs * stmp[k2]) + (sn * e[k2]);
                            e[k2] = (cs * e[k2]) - (sn * stmp[k2]);
                            g = sn * stmp[k2 + 1];
                            stmp[k2 + 1] = cs * stmp[k2 + 1];
                            if (computeVectors)
                            {
                                for (i2 = 0; i2 < n; i2++)
                                {
                                    var z = (cs * vt[(k2 * ldvt) + i2]) + (sn * vt[((k2 + 1) * ldvt) + i2]);
                                    vt[((k2 + 1) * ldvt) + i2] = (cs * vt[((k2 + 1) * ldvt) + i2]) - (sn * vt[(k2 * ldvt) + i2]);
                                    vt[(k2 * ldvt) + i2] = z;
                                }
                            }

                            Drotg(ref f, ref g, out cs, out sn);
                            stmp[k2] = f;
                            f = (cs * e[k2]) + (sn * stmp[k2 + 1]);
                            stmp[k2 + 1] = -(sn * e[k2]) + (cs * stmp[k2 + 1]);
                            g = sn * e[k2 + 1];
                            e[k2 + 1] = cs * e[k2 + 1];
                            if (computeVectors && k2 < m)
                            {
                                for (i2 = 0; i2 < m; i2++)
                                {
                                    var z = (cs * u[(k2 * ldu) + i2]) + (sn * u[((k2 + 1) * ldu) + i2]);
                                    u[((k2 + 1) * ldu) + i2] = (cs * u[((k2 + 1) * ldu) + i2]) - (sn * u[(k2 * ldu) + i2]);
                                    u[(k2 * ldu) + i2] = z;
                                }
                            }
                        }

                        e[p - 2] = f;
                        iter = iter + 1;
                        break;

                    // Convergence
                    case 4:

                        // Make the singular value  positive
                        if (stmp[k] < 0.0)
                        {
                            stmp[k] = -stmp[k];
                            if (computeVectors)
                            {
                                // A part of column "l" of matrix VT from row 0 to end multiply by -1
                                for (i2 = 0; i2 < n; i2++)
                                {
                                    vt[(k * ldvt) + i2] = vt[(k * ldvt) + i2] * -1.0;
                                }
                            }
                        }

                        // Order the singular value.
                        while (k != mn - 1)
                        {
                            if (stmp[k] >= stmp[k + 1])
                            {
                                break;
                            }

                            t2 = stmp[k];
                            stmp[k] = stmp[k + 1];
                            stmp[k + 1] = t2;
                            if (computeVectors && k < n)
                            {
                                // Swap columns l, l + 1
                                for (i2 = 0; i2 < n; i2++)
                                {
                                    (vt[(k * ldvt) + i2], vt[((k + 1) * ldvt) + i2]) = (vt[((k + 1) * ldvt) + i2], vt[(k * ldvt) + i2]);
                                }
                            }

                            if (computeVectors && k < m)
                            {
                                // Swap columns l, l + 1
                                for (i2 = 0; i2 < m; i2++)
                                {
                                    (u[(k * ldu) + i2], u[((k + 1) * ldu) + i2]) = (u[((k + 1) * ldu) + i2], u[(k * ldu) + i2]);
                                }
                            }

                            k = k + 1;
                        }

                        iter = 0;
                        p = p - 1;
                        break;
                }
            }

            if (computeVectors)
            {
                // Finally transpose "v" to get "vt" matrix
                for (i2 = 0; i2 < n; i2++)
                {
                    for (j2 = 0; j2 < i2; j2++)
                    {
                        (vt[(j2 * ldvt) + i2], vt[(i2 * ldvt) + j2]) = (vt[(i2 * ldvt) + j2], vt[(j2 * ldvt) + i2]);
                    }
                }
            }

            // Copy stemp to s with size adjustment. We are using ported copy of linpack's svd code and it uses
            // a singular vector of length rows+1 when rows < columns. The last element is not used and needs to be removed.
            // We should port lapack's svd routine to remove this problem.
            Buffer.MemoryCopy(stmp, s, Math.Min(m, n) * sizeof(double), Math.Min(m, n) * sizeof(double));
        }

        public static unsafe void Svd(int rowsA, int columnsA, Complex* a, int lda, Complex* s, Complex* u, int ldu, Complex* vt, int ldvt)
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

        private static unsafe void SvdCore(int m, int n, Complex* a, int lda, Complex* s, Complex* u, int ldu, Complex* vt, int ldvt, Complex* work, Complex* e, Complex* stmp)
        {
            const int maxiter = 1000;

            // Reduce A to bidiagonal form, storing the diagonal elements in s and the super-diagonal elements in e.
            var nct = Math.Min(m - 1, n);
            var nrt = Math.Max(0, Math.Min(n - 2, m));
            var kmax = Math.Max(nct, nrt);
            for (var k = 0; k < kmax; k++)
            {
                var kp1 = k + 1;
                var aColk = a + lda * k;

                if (k < nct)
                {
                    // Compute the transformation for the k-th column and place the k-th diagonal in s[k].
                    // Compute 2-norm of k-th column.
                    stmp[k] = SvdNorm(m - k, aColk + k);

                    if (stmp[k] != Complex.Zero)
                    {
                        if (aColk[k] != Complex.Zero)
                        {
                            stmp[k] = ChangeArgument(stmp[k], aColk[k]);
                        }

                        SvdDivInplace(m - k, aColk + k, stmp[k]);

                        aColk[k] += 1.0;
                    }

                    stmp[k] = -stmp[k];
                }

                for (var j = kp1; j < n; j++)
                {
                    var aColj = a + lda * j;

                    if (k < nct && stmp[k] != Complex.Zero)
                    {
                        // Apply the transformation.
                        var t = -SvdDot(m - k, aColk + k, aColj + k) / aColk[k];
                        SvdMulAdd(m - k, aColk + k, t, aColj + k);
                    }

                    // Place the k-th row of A into e for the subsequent calculation of the row transformation.
                    e[j] = aColj[k].Conjugate();
                }

                if (u != null && k < nct)
                {
                    // Place the transformation in U for subsequent back multiplication.
                    var uColk = u + ldu * k;
                    var copyLength = sizeof(Complex) * (m - k);
                    Buffer.MemoryCopy(aColk + k, uColk + k, copyLength, copyLength);
                }

                if (k < nrt)
                {
                    // Compute the k-th row transformation and place the k-th super-diagonal in e[k].
                    // Compute 2-norm.
                    e[k] = SvdNorm(n - kp1, e + kp1);

                    if (e[k] != Complex.Zero)
                    {
                        if (e[kp1] != Complex.Zero)
                        {
                            e[k] = ChangeArgument(e[k], e[kp1]);
                        }

                        SvdDivInplace(n - kp1, e + kp1, e[k]);

                        e[kp1] += Complex.One;
                    }

                    e[k] = new Complex(-e[k].Real, e[k].Imaginary);

                    if (kp1 < m && e[k] != Complex.Zero)
                    {
                        // Apply the transformation.
                        new Span<Complex>(work + kp1, m - kp1).Clear();

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            SvdMulAdd(m - kp1, aColj + kp1, e[j], work + kp1);
                        }

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            SvdMulAdd(m - kp1, work + kp1, (-e[j] / e[kp1]).Conjugate(), aColj + kp1);
                        }
                    }

                    if (vt != null)
                    {
                        // Place the transformation in V for subsequent back multiplication.
                        var vtColk = vt + ldvt * k;
                        vtColk[k] = Complex.Zero;
                        var copyLength = sizeof(Complex) * (n - kp1);
                        Buffer.MemoryCopy(e + kp1, vtColk + kp1, copyLength, copyLength);
                    }
                }
            }

            // Set up the final bidiagonal matrix or order p.
            var p = Math.Min(n, m + 1);
            if (nct < n)
            {
                stmp[nct] = a[(nct * lda) + nct];
            }
            if (m < p)
            {
                stmp[p - 1] = Complex.Zero;
            }
            if (nrt + 1 < p)
            {
                e[nrt] = a[((p - 1) * lda) + nrt];
            }

            e[p - 1] = Complex.Zero;

            // If required, generate U.
            if (u != null)
            {
                for (var j = nct; j < m; j++)
                {
                    var uColj = u + ldu * j;
                    new Span<Complex>(uColj, m).Clear();
                    uColj[j] = Complex.One;
                }

                for (var k = nct - 1; k >= 0; k--)
                {
                    var uColk = u + ldu * k;

                    if (stmp[k] != Complex.Zero)
                    {
                        for (var j = k + 1; j < m; j++)
                        {
                            var uColj = u + ldu * j;
                            var t = -SvdDot(m - k, uColk + k, uColj + k) / uColk[k];
                            SvdMulAdd(m - k, uColk + k, t, uColj + k);
                        }

                        SvdFlipSign(m - k, uColk + k);

                        uColk[k] += Complex.One;
                        new Span<Complex>(uColk, k).Clear();
                    }
                    else
                    {
                        new Span<Complex>(uColk, m).Clear();
                        uColk[k] = Complex.One;
                    }
                }
            }

            // If required, generate V.
            if (vt != null)
            {
                for (var k = n - 1; k >= 0; k--)
                {
                    var kp1 = k + 1;
                    var vtColk = vt + ldvt * k;

                    if (k < nrt && e[k] != Complex.Zero)
                    {
                        for (var j = kp1; j < n; j++)
                        {
                            var vtColj = vt + ldvt * j;
                            var t = -SvdDot(n - kp1, vtColk + kp1, vtColj + kp1) / vtColk[kp1];
                            SvdMulAdd(n - k, vtColk + k, t, vtColj + k);
                        }
                    }

                    new Span<Complex>(vtColk, n).Clear();
                    vtColk[k] = Complex.One;
                }
            }

            for (var i = 0; i < p; i++)
            {
                if (stmp[i] != Complex.Zero)
                {
                    var t = stmp[i].Magnitude;
                    var r = stmp[i] / t;
                    stmp[i] = t;
                    if (i < p - 1)
                    {
                        e[i] /= r;
                    }

                    if (u != null)
                    {
                        SvdMulInplace(m, u + ldu * i, r);
                    }
                }

                if (i == p - 1)
                {
                    break;
                }

                if (e[i] != Complex.Zero)
                {
                    var t = e[i].Magnitude;
                    var r = t / e[i];
                    e[i] = t;
                    stmp[i + 1] = stmp[i + 1] * r;

                    if (vt != null)
                    {
                        SvdMulInplace(n, vt + ldvt * (i + 1), r);
                    }
                }
            }

            // Main iteration loop for the singular values.
            var pp = p - 1;
            var iter = 0;
            var eps = Math.Pow(2.0, -52.0);
            while (p > 0)
            {
                // Quit if all the singular values have been found.
                // If too many iterations have been performed throw exception.
                if (iter >= maxiter)
                {
                    throw new Exception();
                }

                // Here is where a test for too many iterations would go.
                // This section of the program inspects for
                // negligible elements in the s and e arrays.  On
                // completion the variables kase and k are set as follows.
                // kase = 1     if s(p) and e[k-1] are negligible and k<p
                // kase = 2     if s(k) is negligible and k<p
                // kase = 3     if e[k-1] is negligible, k<p, and s(k), ..., s(p) are not negligible (qr step).
                // kase = 4     if e(p-1) is negligible (convergence).
                int k;
                for (k = p - 2; k >= 0; k--)
                {
                    if (e[k].Magnitude <= eps * (stmp[k].Magnitude + stmp[k + 1].Magnitude))
                    {
                        e[k] = Complex.Zero;
                        break;
                    }
                }

                int kase;
                if (k == p - 2)
                {
                    kase = 4;
                }
                else
                {
                    int ks;
                    for (ks = p - 1; ks > k; ks--)
                    {
                        var t = 0.0;
                        if (ks != p - 1)
                        {
                            t += e[ks].Magnitude;
                        }
                        if (ks != k + 1)
                        {
                            t += e[ks - 1].Magnitude;
                        }
                        if (stmp[ks].Magnitude <= eps * t)
                        {
                            stmp[ks] = Complex.Zero;
                            break;
                        }
                    }

                    if (ks == k)
                    {
                        kase = 3;
                    }
                    else if (ks == p - 1)
                    {
                        kase = 1;
                    }
                    else
                    {
                        kase = 2;
                        k = ks;
                    }
                }

                k++;

                // Perform the task indicated by kase.
                double f;
                double cs;
                double sn;
                switch (kase)
                {
                    // Deflate negligible s(p).
                    case 1:
                        f = e[p - 2].Real;
                        e[p - 2] = Complex.Zero;
                        for (var j = k; j < p - 1; j++)
                        {
                            var l = p - 2 - j + k;
                            var t = stmp[l].Real;
                            Drotg(ref t, ref f, out cs, out sn);
                            stmp[l] = t;
                            if (l != k)
                            {
                                f = -sn * e[l - 1].Real;
                                e[l - 1] = cs * e[l - 1];
                            }

                            if (vt != null)
                            {
                                var vtColl = vt + ldvt * l;
                                var vtColpm1 = vt + ldvt * (p - 1);
                                for (var i = 0; i < n; i++)
                                {
                                    var z = cs * vtColl[i] + sn * vtColpm1[i];
                                    vtColpm1[i] = cs * vtColpm1[i] - sn * vtColl[i];
                                    vtColl[i] = z;
                                }
                            }
                        }

                        break;

                    // Split at negligible s(k).
                    case 2:
                        f = e[k - 1].Real;
                        e[k - 1] = 0.0;
                        for (var j = k; j < p; j++)
                        {
                            var t = stmp[j].Real;
                            Drotg(ref t, ref f, out cs, out sn);
                            stmp[j] = t;
                            f = -sn * e[j].Real;
                            e[j] = cs * e[j];

                            if (u != null)
                            {
                                var uColj = u + ldu * j;
                                var uColkm1 = u + ldu * (k - 1);
                                for (var i = 0; i < m; i++)
                                {
                                    var z = (cs * uColj[i]) + (sn * uColkm1[i]);
                                    uColkm1[i] = (cs * uColkm1[i]) - (sn * uColj[i]);
                                    uColj[i] = z;
                                }
                            }
                        }

                        break;

                    // Perform one qr step.
                    case 3:
                        // Calculate the shift.
                        var scale = 0.0;
                        scale = Math.Max(scale, stmp[p - 1].Magnitude);
                        scale = Math.Max(scale, stmp[p - 2].Magnitude);
                        scale = Math.Max(scale, e[p - 2].Magnitude);
                        scale = Math.Max(scale, stmp[k].Magnitude);
                        scale = Math.Max(scale, e[k].Magnitude);
                        var sp = stmp[p - 1].Real / scale;
                        var spm1 = stmp[p - 2].Real / scale;
                        var epm1 = e[p - 2].Real / scale;
                        var sk = stmp[k].Real / scale;
                        var ek = e[k].Real / scale;
                        var b = (((spm1 + sp) * (spm1 - sp)) + (epm1 * epm1)) / 2.0;
                        var c = (sp * epm1) * (sp * epm1);
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

                        f = ((sk + sp) * (sk - sp)) + shift;
                        var g = sk * ek;

                        // Chase zeros.
                        for (var j = k; j < p - 1; j++)
                        {
                            Drotg(ref f, ref g, out cs, out sn);
                            if (j != k)
                            {
                                e[j - 1] = f;
                            }

                            f = cs * stmp[j].Real + sn * e[j].Real;
                            e[j] = cs * e[j] - sn * stmp[j];
                            g = sn * stmp[j + 1].Real;
                            stmp[j + 1] = cs * stmp[j + 1];

                            if (vt != null)
                            {
                                for (var i = 0; i < n; i++)
                                {
                                    var vtColj = vt + ldvt * j;
                                    var vtColjp1 = vt + ldvt * (j + 1);
                                    var z = cs * vtColj[i] + sn * vtColjp1[i];
                                    vtColjp1[i] = cs * vtColjp1[i] - sn * vtColj[i];
                                    vtColj[i] = z;
                                }
                            }

                            Drotg(ref f, ref g, out cs, out sn);
                            stmp[j] = f;
                            f = cs * e[j].Real + sn * stmp[j + 1].Real;
                            stmp[j + 1] = -(sn * e[j]) + (cs * stmp[j + 1]);
                            g = sn * e[j + 1].Real;
                            e[j + 1] = cs * e[j + 1];

                            if (u != null && j < m)
                            {
                                for (var i = 0; i < m; i++)
                                {
                                    var uColj = u + ldu * j;
                                    var uColjp1 = u + ldu * (j + 1);
                                    var z = cs * uColj[i] + sn * uColjp1[i];
                                    uColjp1[i] = cs * uColjp1[i] - sn * uColj[i];
                                    uColj[i] = z;
                                }
                            }
                        }

                        e[p - 2] = f;
                        iter++;

                        break;

                    // Convergence.
                    case 4:
                        // Make the singular value positive.
                        if (stmp[k].Real < 0.0)
                        {
                            stmp[k] = -stmp[k];

                            if (vt != null)
                            {
                                SvdFlipSign(n, vt + ldvt * k);
                            }
                        }

                        // Order the singular values.
                        while (k < pp)
                        {
                            if (stmp[k].Real >= stmp[k + 1].Real)
                            {
                                break;
                            }

                            (stmp[k], stmp[k + 1]) = (stmp[k + 1], stmp[k]);

                            if (vt != null && k < n)
                            {
                                var vtColk = vt + ldvt * k;
                                var vtColkp1 = vt + ldvt * (k + 1);
                                for (var i = 0; i < n; i++)
                                {
                                    (vtColk[i], vtColkp1[i]) = (vtColkp1[i], vtColk[i]);
                                }
                            }

                            if (u != null && k < m)
                            {
                                var uColk = u + ldu * k;
                                var uColkp1 = u + ldu * (k + 1);
                                for (var i = 0; i < m; i++)
                                {
                                    (uColk[i], uColkp1[i]) = (uColkp1[i], uColk[i]);
                                }
                            }

                            k++;
                        }

                        iter = 0;
                        p = p - 1;

                        break;
                }
            }

            if (vt != null)
            {
                for (var j = 0; j < n; j++)
                {
                    var vtColj = vt + ldvt * j;

                    for (var i = 0; i <= j; i++)
                    {
                        if (i == j)
                        {
                            vtColj[j] = vtColj[j].Conjugate();
                        }
                        else
                        {
                            var vtColi = vt + ldvt * i;
                            var t1 = vtColj[i];
                            var t2 = vtColi[j];
                            vtColj[i] = t2.Conjugate();
                            vtColi[j] = t1.Conjugate();
                        }
                    }
                }
            }

            new Span<Complex>(stmp, Math.Min(m, n)).CopyTo(new Span<Complex>(s, Math.Min(m, n)));
        }

        private static unsafe double SvdNorm(int n, double* x)
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
                    throw new LinearAlgebraException("An unexpected error occurred.");
            }

            while (n > 0)
            {
                sum += x[0] * x[0] + x[1] * x[1];
                x += 2;
                n -= 2;
            }

            return Math.Sqrt(sum);
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

        private static unsafe void SvdMulInplace(int n, double* x, double y)
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

        private static unsafe double SvdDot(int n, double* x, double* y)
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
                    throw new LinearAlgebraException("An unexpected error occurred.");
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

        private static double SvdFastMagnitude(Complex x)
        {
            return Math.Max(Math.Abs(x.Real), Math.Abs(x.Imaginary));
        }
    }
}
