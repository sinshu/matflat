using Microsoft.VisualBasic;
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

        private static unsafe void SvdCore(int rowsA, int columnsA, double* a, int lda, double* s, double* u, int ldu, double* v, int ldvt, double* work, double* e, double* stemp)
        {
            var computeVectors = true;

            for (var ccc = 0; ccc < columnsA; ccc++)
            {
                new Span<double>(v + ccc * ldvt, columnsA).Clear();
            }

            int i, j, l, lp1;

            double t;

            var ncu = rowsA;

            // Reduce matrix to bidiagonal form, storing the diagonal elements
            // in "s" and the super-diagonal elements in "e".
            var nct = Math.Min(rowsA - 1, columnsA);
            var nrt = Math.Max(0, Math.Min(columnsA - 2, rowsA));
            var lu = Math.Max(nct, nrt);

            for (l = 0; l < lu; l++)
            {
                lp1 = l + 1;
                if (l < nct)
                {
                    // Compute the transformation for the l-th column and
                    // place the l-th diagonal in vector s[l].
                    var sum = 0.0;
                    for (var i1 = l; i1 < rowsA; i1++)
                    {
                        sum += a[(l * rowsA) + i1] * a[(l * rowsA) + i1];
                    }

                    stemp[l] = Math.Sqrt(sum);

                    if (stemp[l] != 0.0)
                    {
                        if (a[(l * rowsA) + l] != 0.0)
                        {
                            stemp[l] = Math.Abs(stemp[l]) * (a[(l * rowsA) + l] / Math.Abs(a[(l * rowsA) + l]));
                        }

                        // A part of column "l" of Matrix A from row "l" to end multiply by 1.0 / s[l]
                        for (i = l; i < rowsA; i++)
                        {
                            a[(l * rowsA) + i] = a[(l * rowsA) + i] * (1.0 / stemp[l]);
                        }

                        a[(l * rowsA) + l] = 1.0 + a[(l * rowsA) + l];
                    }

                    stemp[l] = -stemp[l];
                }

                for (j = lp1; j < columnsA; j++)
                {
                    if (l < nct)
                    {
                        if (stemp[l] != 0.0)
                        {
                            // Apply the transformation.
                            t = 0.0;
                            for (i = l; i < rowsA; i++)
                            {
                                t += a[(j * rowsA) + i] * a[(l * rowsA) + i];
                            }

                            t = -t / a[(l * rowsA) + l];

                            for (var ii = l; ii < rowsA; ii++)
                            {
                                a[(j * rowsA) + ii] += t * a[(l * rowsA) + ii];
                            }
                        }
                    }

                    // Place the l-th row of matrix into "e" for the
                    // subsequent calculation of the row transformation.
                    e[j] = a[(j * rowsA) + l];
                }

                if (computeVectors && l < nct)
                {
                    // Place the transformation in "u" for subsequent back multiplication.
                    for (i = l; i < rowsA; i++)
                    {
                        u[(l * rowsA) + i] = a[(l * rowsA) + i];
                    }
                }

                if (l >= nrt)
                {
                    continue;
                }

                // Compute the l-th row transformation and place the l-th super-diagonal in e(l).
                var enorm = 0.0;
                for (i = lp1; i < columnsA; i++)
                {
                    enorm += e[i] * e[i];
                }

                e[l] = Math.Sqrt(enorm);
                if (e[l] != 0.0)
                {
                    if (e[lp1] != 0.0)
                    {
                        e[l] = Math.Abs(e[l]) * (e[lp1] / Math.Abs(e[lp1]));
                    }

                    // Scale vector "e" from "lp1" by 1.0 / e[l]
                    for (i = lp1; i < columnsA; i++)
                    {
                        e[i] = e[i] * (1.0 / e[l]);
                    }

                    e[lp1] = 1.0 + e[lp1];
                }

                e[l] = -e[l];

                if (lp1 < rowsA && e[l] != 0.0)
                {
                    // Apply the transformation.
                    for (i = lp1; i < rowsA; i++)
                    {
                        work[i] = 0.0;
                    }

                    for (j = lp1; j < columnsA; j++)
                    {
                        for (var ii = lp1; ii < rowsA; ii++)
                        {
                            work[ii] += e[j] * a[(j * rowsA) + ii];
                        }
                    }

                    for (j = lp1; j < columnsA; j++)
                    {
                        var ww = -e[j] / e[lp1];
                        for (var ii = lp1; ii < rowsA; ii++)
                        {
                            a[(j * rowsA) + ii] += ww * work[ii];
                        }
                    }
                }

                if (!computeVectors)
                {
                    continue;
                }

                // Place the transformation in v for subsequent back multiplication.
                for (i = lp1; i < columnsA; i++)
                {
                    v[(l * columnsA) + i] = e[i];
                }
            }

            // Set up the final bidiagonal matrix or order m.
            var m = Math.Min(columnsA, rowsA + 1);
            var nctp1 = nct + 1;
            var nrtp1 = nrt + 1;
            if (nct < columnsA)
            {
                stemp[nctp1 - 1] = a[((nctp1 - 1) * rowsA) + (nctp1 - 1)];
            }

            if (rowsA < m)
            {
                stemp[m - 1] = 0.0;
            }

            if (nrtp1 < m)
            {
                e[nrtp1 - 1] = a[((m - 1) * rowsA) + (nrtp1 - 1)];
            }

            e[m - 1] = 0.0;

            // If required, generate "u".
            if (computeVectors)
            {
                for (j = nctp1 - 1; j < ncu; j++)
                {
                    for (i = 0; i < rowsA; i++)
                    {
                        u[(j * rowsA) + i] = 0.0;
                    }

                    u[(j * rowsA) + j] = 1.0;
                }

                for (l = nct - 1; l >= 0; l--)
                {
                    if (stemp[l] != 0.0)
                    {
                        for (j = l + 1; j < ncu; j++)
                        {
                            t = 0.0;
                            for (i = l; i < rowsA; i++)
                            {
                                t += u[(j * rowsA) + i] * u[(l * rowsA) + i];
                            }

                            t = -t / u[(l * rowsA) + l];

                            for (var ii = l; ii < rowsA; ii++)
                            {
                                u[(j * rowsA) + ii] += t * u[(l * rowsA) + ii];
                            }
                        }

                        // A part of column "l" of matrix A from row "l" to end multiply by -1.0
                        for (i = l; i < rowsA; i++)
                        {
                            u[(l * rowsA) + i] = u[(l * rowsA) + i] * -1.0;
                        }

                        u[(l * rowsA) + l] = 1.0 + u[(l * rowsA) + l];
                        for (i = 0; i < l; i++)
                        {
                            u[(l * rowsA) + i] = 0.0;
                        }
                    }
                    else
                    {
                        for (i = 0; i < rowsA; i++)
                        {
                            u[(l * rowsA) + i] = 0.0;
                        }

                        u[(l * rowsA) + l] = 1.0;
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
                            for (j = lp1; j < columnsA; j++)
                            {
                                t = 0.0;
                                for (i = lp1; i < columnsA; i++)
                                {
                                    t += v[(j * columnsA) + i] * v[(l * columnsA) + i];
                                }

                                t = -t / v[(l * columnsA) + lp1];
                                for (var ii = l; ii < columnsA; ii++)
                                {
                                    v[(j * columnsA) + ii] += t * v[(l * columnsA) + ii];
                                }
                            }
                        }
                    }

                    for (i = 0; i < columnsA; i++)
                    {
                        v[(l * columnsA) + i] = 0.0;
                    }

                    v[(l * columnsA) + l] = 1.0;
                }
            }

            // Transform "s" and "e" so that they are double
            for (i = 0; i < m; i++)
            {
                double r;
                if (stemp[i] != 0.0)
                {
                    t = stemp[i];
                    r = stemp[i] / t;
                    stemp[i] = t;
                    if (i < m - 1)
                    {
                        e[i] = e[i] / r;
                    }

                    if (computeVectors)
                    {
                        // A part of column "i" of matrix U from row 0 to end multiply by r
                        for (j = 0; j < rowsA; j++)
                        {
                            u[(i * rowsA) + j] = u[(i * rowsA) + j] * r;
                        }
                    }
                }

                // Exit
                if (i == m - 1)
                {
                    break;
                }

                if (e[i] == 0.0)
                {
                    continue;
                }

                t = e[i];
                r = t / e[i];
                e[i] = t;
                stemp[i + 1] = stemp[i + 1] * r;
                if (!computeVectors)
                {
                    continue;
                }

                // A part of column "i+1" of matrix VT from row 0 to end multiply by r
                for (j = 0; j < columnsA; j++)
                {
                    v[((i + 1) * columnsA) + j] = v[((i + 1) * columnsA) + j] * r;
                }
            }

            // Main iteration loop for the singular values.
            var mn = m;
            var iter = 0;

            while (m > 0)
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
                for (l = m - 2; l >= 0; l--)
                {
                    test = Math.Abs(stemp[l]) + Math.Abs(stemp[l + 1]);
                    ztest = test + Math.Abs(e[l]);
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
                            test = test + Math.Abs(e[ls]);
                        }

                        if (ls != l + 1)
                        {
                            test = test + Math.Abs(e[ls - 1]);
                        }

                        ztest = test + Math.Abs(stemp[ls]);
                        if (ztest.AlmostEqualRelative(test, 15))
                        {
                            stemp[ls] = 0.0;
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
                        f = e[m - 2];
                        e[m - 2] = 0.0;
                        double t1;
                        for (var kk = l; kk < m - 1; kk++)
                        {
                            k = m - 2 - kk + l;
                            t1 = stemp[k];

                            Drotg(ref t1, ref f, out cs, out sn);
                            stemp[k] = t1;
                            if (k != l)
                            {
                                f = -sn * e[k - 1];
                                e[k - 1] = cs * e[k - 1];
                            }

                            if (computeVectors)
                            {
                                // Rotate
                                for (i = 0; i < columnsA; i++)
                                {
                                    var z = (cs * v[(k * columnsA) + i]) + (sn * v[((m - 1) * columnsA) + i]);
                                    v[((m - 1) * columnsA) + i] = (cs * v[((m - 1) * columnsA) + i]) - (sn * v[(k * columnsA) + i]);
                                    v[(k * columnsA) + i] = z;
                                }
                            }
                        }

                        break;

                    // Split at negligible s[l].
                    case 2:
                        f = e[l - 1];
                        e[l - 1] = 0.0;
                        for (k = l; k < m; k++)
                        {
                            t1 = stemp[k];
                            Drotg(ref t1, ref f, out cs, out sn);
                            stemp[k] = t1;
                            f = -sn * e[k];
                            e[k] = cs * e[k];
                            if (computeVectors)
                            {
                                // Rotate
                                for (i = 0; i < rowsA; i++)
                                {
                                    var z = (cs * u[(k * rowsA) + i]) + (sn * u[((l - 1) * rowsA) + i]);
                                    u[((l - 1) * rowsA) + i] = (cs * u[((l - 1) * rowsA) + i]) - (sn * u[(k * rowsA) + i]);
                                    u[(k * rowsA) + i] = z;
                                }
                            }
                        }

                        break;

                    // Perform one qr step.
                    case 3:

                        // calculate the shift.
                        var scale = 0.0;
                        scale = Math.Max(scale, Math.Abs(stemp[m - 1]));
                        scale = Math.Max(scale, Math.Abs(stemp[m - 2]));
                        scale = Math.Max(scale, Math.Abs(e[m - 2]));
                        scale = Math.Max(scale, Math.Abs(stemp[l]));
                        scale = Math.Max(scale, Math.Abs(e[l]));
                        var sm = stemp[m - 1] / scale;
                        var smm1 = stemp[m - 2] / scale;
                        var emm1 = e[m - 2] / scale;
                        var sl = stemp[l] / scale;
                        var el = e[l] / scale;
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

                            f = (cs * stemp[k]) + (sn * e[k]);
                            e[k] = (cs * e[k]) - (sn * stemp[k]);
                            g = sn * stemp[k + 1];
                            stemp[k + 1] = cs * stemp[k + 1];
                            if (computeVectors)
                            {
                                for (i = 0; i < columnsA; i++)
                                {
                                    var z = (cs * v[(k * columnsA) + i]) + (sn * v[((k + 1) * columnsA) + i]);
                                    v[((k + 1) * columnsA) + i] = (cs * v[((k + 1) * columnsA) + i]) - (sn * v[(k * columnsA) + i]);
                                    v[(k * columnsA) + i] = z;
                                }
                            }

                            Drotg(ref f, ref g, out cs, out sn);
                            stemp[k] = f;
                            f = (cs * e[k]) + (sn * stemp[k + 1]);
                            stemp[k + 1] = -(sn * e[k]) + (cs * stemp[k + 1]);
                            g = sn * e[k + 1];
                            e[k + 1] = cs * e[k + 1];
                            if (computeVectors && k < rowsA)
                            {
                                for (i = 0; i < rowsA; i++)
                                {
                                    var z = (cs * u[(k * rowsA) + i]) + (sn * u[((k + 1) * rowsA) + i]);
                                    u[((k + 1) * rowsA) + i] = (cs * u[((k + 1) * rowsA) + i]) - (sn * u[(k * rowsA) + i]);
                                    u[(k * rowsA) + i] = z;
                                }
                            }
                        }

                        e[m - 2] = f;
                        iter = iter + 1;
                        break;

                    // Convergence
                    case 4:

                        // Make the singular value  positive
                        if (stemp[l] < 0.0)
                        {
                            stemp[l] = -stemp[l];
                            if (computeVectors)
                            {
                                // A part of column "l" of matrix VT from row 0 to end multiply by -1
                                for (i = 0; i < columnsA; i++)
                                {
                                    v[(l * columnsA) + i] = v[(l * columnsA) + i] * -1.0;
                                }
                            }
                        }

                        // Order the singular value.
                        while (l != mn - 1)
                        {
                            if (stemp[l] >= stemp[l + 1])
                            {
                                break;
                            }

                            t = stemp[l];
                            stemp[l] = stemp[l + 1];
                            stemp[l + 1] = t;
                            if (computeVectors && l < columnsA)
                            {
                                // Swap columns l, l + 1
                                for (i = 0; i < columnsA; i++)
                                {
                                    (v[(l * columnsA) + i], v[((l + 1) * columnsA) + i]) = (v[((l + 1) * columnsA) + i], v[(l * columnsA) + i]);
                                }
                            }

                            if (computeVectors && l < rowsA)
                            {
                                // Swap columns l, l + 1
                                for (i = 0; i < rowsA; i++)
                                {
                                    (u[(l * rowsA) + i], u[((l + 1) * rowsA) + i]) = (u[((l + 1) * rowsA) + i], u[(l * rowsA) + i]);
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
                for (i = 0; i < columnsA; i++)
                {
                    for (j = 0; j < i; j++)
                    {
                        (v[(j * columnsA) + i], v[(i * columnsA) + j]) = (v[(i * columnsA) + j], v[(j * columnsA) + i]);
                    }
                }
            }

            // Copy stemp to s with size adjustment. We are using ported copy of linpack's svd code and it uses
            // a singular vector of length rows+1 when rows < columns. The last element is not used and needs to be removed.
            // We should port lapack's svd routine to remove this problem.
            Buffer.MemoryCopy(stemp, s, Math.Min(rowsA, columnsA) * sizeof(double), Math.Min(rowsA, columnsA) * sizeof(double));
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

                    if (k < nrt)
                    {
                        if (e[k] != Complex.Zero)
                        {
                            for (var j = kp1; j < n; j++)
                            {
                                var vtColj = vt + ldvt * j;
                                var t = -SvdDot(n - kp1, vtColk + kp1, vtColj + kp1) / vtColk[kp1];
                                SvdMulAdd(n - k, vtColk + k, t, vtColj + k);
                            }
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

        private static double SvdFastMagnitude(Complex x)
        {
            return Math.Max(Math.Abs(x.Real), Math.Abs(x.Imaginary));
        }
    }
}
