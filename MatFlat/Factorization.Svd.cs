using System;
using System.Buffers;
using System.Numerics;

namespace MatFlat
{
    public static partial class Factorization
    {
        /// <summary>
        /// computes the singular value decomposition (SVD) of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the contents of A are destroyed.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="s">
        /// On exit, the singular values are stored.
        /// </param>
        /// <param name="u">
        /// On exit, U contains the M-by-M orthogonal matrix corresponding to the left singular vectors.
        /// If null, the left singular vectors are not computed.
        /// </param>
        /// <param name="ldu">
        /// The leading dimension of the array U.
        /// </param>
        /// <param name="vt">
        /// On exit, VT contains the N-by-N orthogonal matrix corresponding to the right singular vectors.
        /// If null, the right singular vectors are not computed.
        /// </param>
        /// <param name="ldvt">
        /// The leading dimension of the array VT.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The solution did not converge.
        /// </exception>
        public static unsafe void Svd(int m, int n, float* a, int lda, float* s, float* u, int ldu, float* vt, int ldvt)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (s == null)
            {
                throw new ArgumentNullException(nameof(s));
            }

            if (u != null)
            {
                if (ldu < m)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldu));
                }
            }

            if (vt != null)
            {
                if (ldvt < n)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldvt));
                }
            }

            var bufferLength = m + n + Math.Min(m + 1, n);
            var buffer = ArrayPool<float>.Shared.Rent(bufferLength);
            try
            {
                fixed (float* p = buffer)
                {
                    var work = p;
                    var e = work + m;
                    var stmp = e + n;
                    SvdCore(m, n, a, lda, s, u, ldu, vt, ldvt, work, e, stmp);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }

        /// <summary>
        /// computes the singular value decomposition (SVD) of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the contents of A are destroyed.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="s">
        /// On exit, the singular values are stored.
        /// </param>
        /// <param name="u">
        /// On exit, U contains the M-by-M orthogonal matrix corresponding to the left singular vectors.
        /// If null, the left singular vectors are not computed.
        /// </param>
        /// <param name="ldu">
        /// The leading dimension of the array U.
        /// </param>
        /// <param name="vt">
        /// On exit, VT contains the N-by-N orthogonal matrix corresponding to the right singular vectors.
        /// If null, the right singular vectors are not computed.
        /// </param>
        /// <param name="ldvt">
        /// The leading dimension of the array VT.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The solution did not converge.
        /// </exception>
        public static unsafe void Svd(int m, int n, double* a, int lda, double* s, double* u, int ldu, double* vt, int ldvt)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (s == null)
            {
                throw new ArgumentNullException(nameof(s));
            }

            if (u != null)
            {
                if (ldu < m)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldu));
                }
            }

            if (vt != null)
            {
                if (ldvt < n)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldvt));
                }
            }

            var bufferLength = m + n + Math.Min(m + 1, n);
            var buffer = ArrayPool<double>.Shared.Rent(bufferLength);
            try
            {
                fixed (double* p = buffer)
                {
                    var work = p;
                    var e = work + m;
                    var stmp = e + n;
                    SvdCore(m, n, a, lda, s, u, ldu, vt, ldvt, work, e, stmp);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(buffer);
            }
        }

        /// <summary>
        /// computes the singular value decomposition (SVD) of a general M-by-N matrix A.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// <para>
        /// On entry, the M-by-N matrix to be factored.
        /// </para>
        /// <para>
        /// On exit, the contents of A are destroyed.
        /// </para>
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="s">
        /// On exit, the singular values are stored.
        /// </param>
        /// <param name="u">
        /// On exit, U contains the M-by-M orthogonal matrix corresponding to the left singular vectors.
        /// If null, the left singular vectors are not computed.
        /// </param>
        /// <param name="ldu">
        /// The leading dimension of the array U.
        /// </param>
        /// <param name="vt">
        /// On exit, VT contains the N-by-N orthogonal matrix corresponding to the right singular vectors.
        /// If null, the right singular vectors are not computed.
        /// </param>
        /// <param name="ldvt">
        /// The leading dimension of the array VT.
        /// </param>
        /// <exception cref="MatrixFactorizationException">
        /// The solution did not converge.
        /// </exception>
        public static unsafe void Svd(int m, int n, Complex* a, int lda, double* s, Complex* u, int ldu, Complex* vt, int ldvt)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (s == null)
            {
                throw new ArgumentNullException(nameof(s));
            }

            if (u != null)
            {
                if (ldu < m)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldu));
                }
            }

            if (vt != null)
            {
                if (ldvt < n)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldvt));
                }
            }

            var bufferLength = m + n + Math.Min(m + 1, n);
            var buffer = ArrayPool<Complex>.Shared.Rent(bufferLength);
            try
            {
                fixed (Complex* p = buffer)
                {
                    var work = p;
                    var e = work + m;
                    var stmp = e + n;
                    SvdCore(m, n, a, lda, s, u, ldu, vt, ldvt, work, e, stmp);
                }
            }
            finally
            {
                ArrayPool<Complex>.Shared.Return(buffer);
            }
        }

        private static unsafe void SvdCore(int m, int n, float* a, int lda, float* s, float* u, int ldu, float* vt, int ldvt, float* work, float* e, float* stmp)
        {
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
                    stmp[k] = (float)Internals.Norm(m - k, aColk + k);

                    if (stmp[k] != 0.0F)
                    {
                        if (aColk[k] != 0.0F)
                        {
                            stmp[k] = Math.Abs(stmp[k]) * (aColk[k] / Math.Abs(aColk[k]));
                        }

                        Internals.DivInplace(m - k, aColk + k, stmp[k]);

                        aColk[k] += 1.0F;
                    }

                    stmp[k] = -stmp[k];
                }

                for (var j = kp1; j < n; j++)
                {
                    var aColj = a + lda * j;

                    if (k < nct && stmp[k] != 0.0)
                    {
                        // Apply the transformation.
                        var t = -(float)(Internals.Dot(m - k, aColk + k, aColj + k) / aColk[k]);
                        Internals.MulAdd(m - k, aColk + k, t, aColj + k);
                    }

                    // Place the k-th row of A into e for the subsequent calculation of the row transformation.
                    e[j] = aColj[k];
                }

                if (u != null && k < nct)
                {
                    // Place the transformation in U for subsequent back multiplication.
                    var uColk = u + ldu * k;
                    var copyLength = sizeof(float) * (m - k);
                    Buffer.MemoryCopy(aColk + k, uColk + k, copyLength, copyLength);
                }

                if (k < nrt)
                {
                    // Compute the k-th row transformation and place the k-th super-diagonal in e[k].
                    // Compute 2-norm.
                    e[k] = (float)Internals.Norm(n - kp1, e + kp1);

                    if (e[k] != 0.0)
                    {
                        if (e[kp1] != 0.0)
                        {
                            e[k] = Math.Abs(e[k]) * (e[kp1] / Math.Abs(e[kp1]));
                        }

                        Internals.DivInplace(n - kp1, e + kp1, e[k]);

                        e[kp1] += 1.0F;
                    }

                    e[k] = -e[k];

                    if (kp1 < m && e[k] != 0.0F)
                    {
                        // Apply the transformation.
                        new Span<float>(work + kp1, m - kp1).Clear();

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            Internals.MulAdd(m - kp1, aColj + kp1, e[j], work + kp1);
                        }

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            Internals.MulAdd(m - kp1, work + kp1, -e[j] / e[kp1], aColj + kp1);
                        }
                    }

                    if (vt != null)
                    {
                        // Place the transformation in V for subsequent back multiplication.
                        var vtColk = vt + ldvt * k;
                        vtColk[k] = 0.0F;
                        var copyLength = sizeof(float) * (n - kp1);
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
                stmp[p - 1] = 0.0F;
            }
            if (nrt + 1 < p)
            {
                e[nrt] = a[((p - 1) * lda) + nrt];
            }

            e[p - 1] = 0.0F;

            // If required, generate U.
            if (u != null)
            {
                for (var j = nct; j < m; j++)
                {
                    var uColj = u + ldu * j;
                    new Span<float>(uColj, m).Clear();
                    uColj[j] = 1.0F;
                }

                for (var k = nct - 1; k >= 0; k--)
                {
                    var uColk = u + ldu * k;

                    if (stmp[k] != 0.0)
                    {
                        for (var j = k + 1; j < m; j++)
                        {
                            var uColj = u + ldu * j;
                            var t = -(float)(Internals.Dot(m - k, uColk + k, uColj + k) / uColk[k]);
                            Internals.MulAdd(m - k, uColk + k, t, uColj + k);
                        }

                        Internals.FlipSign(m - k, uColk + k);

                        uColk[k] += 1.0F;
                        new Span<float>(uColk, k).Clear();
                    }
                    else
                    {
                        new Span<float>(uColk, m).Clear();
                        uColk[k] = 1.0F;
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
                            var t = -(float)(Internals.Dot(n - kp1, vtColk + kp1, vtColj + kp1) / vtColk[kp1]);
                            Internals.MulAdd(n - k, vtColk + k, t, vtColj + k);
                        }
                    }

                    new Span<float>(vtColk, n).Clear();
                    vtColk[k] = 1.0F;
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
                        Internals.MulInplace(m, u + ldu * i, r);
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
                        Internals.MulInplace(n, vt + ldvt * (i + 1), r);
                    }
                }
            }

            // Main iteration loop for the singular values.
            var pp = p - 1;
            var iter = 0;
            var eps = MathF.Pow(2.0F, -23.0F);
            while (p > 0)
            {
                // Quit if all the singular values have been found.
                // If too many iterations have been performed throw exception.
                if (iter >= 1000)
                {
                    throw new MatrixFactorizationException("SVD failed. The solution did not converge.");
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
                    if (Math.Abs(e[k]) <= eps * (Math.Abs(stmp[k]) + Math.Abs(stmp[k + 1])))
                    {
                        e[k] = 0.0F;
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
                            t += Math.Abs(e[ks]);
                        }
                        if (ks != k + 1)
                        {
                            t += Math.Abs(e[ks - 1]);
                        }
                        if (Math.Abs(stmp[ks]) <= eps * t)
                        {
                            stmp[ks] = 0.0F;
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
                float f;
                float cs;
                float sn;
                switch (kase)
                {
                    // Deflate negligible s(p).
                    case 1:
                        f = e[p - 2];
                        e[p - 2] = 0.0F;
                        for (var j = k; j < p - 1; j++)
                        {
                            var l = p - 2 - j + k;
                            var t = stmp[l];
                            Internals.Srotg(ref t, ref f, out cs, out sn);
                            stmp[l] = t;
                            if (l != k)
                            {
                                f = -sn * e[l - 1];
                                e[l - 1] *= cs;
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
                        f = e[k - 1];
                        e[k - 1] = 0.0F;
                        for (var j = k; j < p; j++)
                        {
                            var t = stmp[j];
                            Internals.Srotg(ref t, ref f, out cs, out sn);
                            stmp[j] = t;
                            f = -sn * e[j];
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
                        var scale = 0.0F;
                        scale = MathF.Max(scale, MathF.Abs(stmp[p - 1]));
                        scale = MathF.Max(scale, MathF.Abs(stmp[p - 2]));
                        scale = MathF.Max(scale, MathF.Abs(e[p - 2]));
                        scale = MathF.Max(scale, MathF.Abs(stmp[k]));
                        scale = MathF.Max(scale, MathF.Abs(e[k]));
                        var sp = stmp[p - 1] / scale;
                        var spm1 = stmp[p - 2] / scale;
                        var epm1 = e[p - 2] / scale;
                        var sk = stmp[k] / scale;
                        var ek = e[k] / scale;
                        var b = (((spm1 + sp) * (spm1 - sp)) + (epm1 * epm1)) / 2.0F;
                        var c = (sp * epm1) * (sp * epm1);
                        var shift = 0.0F;
                        if (b != 0.0 || c != 0.0)
                        {
                            shift = MathF.Sqrt((b * b) + c);
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
                            Internals.Srotg(ref f, ref g, out cs, out sn);
                            if (j != k)
                            {
                                e[j - 1] = f;
                            }

                            f = (cs * stmp[j]) + (sn * e[j]);
                            e[j] = (cs * e[j]) - (sn * stmp[j]);
                            g = sn * stmp[j + 1];
                            stmp[j + 1] *= cs;

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

                            Internals.Srotg(ref f, ref g, out cs, out sn);
                            stmp[j] = f;
                            f = (cs * e[j]) + (sn * stmp[j + 1]);
                            stmp[j + 1] = -(sn * e[j]) + (cs * stmp[j + 1]);
                            g = sn * e[j + 1];
                            e[j + 1] *= cs;

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
                        iter = iter++;

                        break;

                    // Convergence.
                    case 4:

                        // Make the singular value positive.
                        if (stmp[k] < 0.0)
                        {
                            stmp[k] = -stmp[k];

                            if (vt != null)
                            {
                                if (vt != null)
                                {
                                    Internals.FlipSign(n, vt + ldvt * k);
                                }
                            }
                        }

                        // Order the singular value.
                        while (k < pp)
                        {
                            if (stmp[k] >= stmp[k + 1])
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
                        p--;

                        break;
                }
            }

            if (vt != null)
            {
                var vtColj = vt;
                for (var j = 0; j < n; j++)
                {
                    var vtColi = vt;
                    for (var i = 0; i < j; i++)
                    {
                        (vtColj[i], vtColi[j]) = (vtColi[j], vtColj[i]);
                        vtColi += ldvt;
                    }
                    vtColj += ldvt;
                }
            }

            var copySize = sizeof(float) * Math.Min(m, n);
            Buffer.MemoryCopy(stmp, s, copySize, copySize);
        }

        private static unsafe void SvdCore(int m, int n, double* a, int lda, double* s, double* u, int ldu, double* vt, int ldvt, double* work, double* e, double* stmp)
        {
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
                    stmp[k] = Internals.Norm(m - k, aColk + k);

                    if (stmp[k] != 0.0)
                    {
                        if (aColk[k] != 0.0)
                        {
                            stmp[k] = Math.Abs(stmp[k]) * (aColk[k] / Math.Abs(aColk[k]));
                        }

                        Internals.DivInplace(m - k, aColk + k, stmp[k]);

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
                        var t = -Internals.Dot(m - k, aColk + k, aColj + k) / aColk[k];
                        Internals.MulAdd(m - k, aColk + k, t, aColj + k);
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
                    e[k] = Internals.Norm(n - kp1, e + kp1);

                    if (e[k] != 0.0)
                    {
                        if (e[kp1] != 0.0)
                        {
                            e[k] = Math.Abs(e[k]) * (e[kp1] / Math.Abs(e[kp1]));
                        }

                        Internals.DivInplace(n - kp1, e + kp1, e[k]);

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
                            Internals.MulAdd(m - kp1, aColj + kp1, e[j], work + kp1);
                        }

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            Internals.MulAdd(m - kp1, work + kp1, -e[j] / e[kp1], aColj + kp1);
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
                            var t = -Internals.Dot(m - k, uColk + k, uColj + k) / uColk[k];
                            Internals.MulAdd(m - k, uColk + k, t, uColj + k);
                        }

                        Internals.FlipSign(m - k, uColk + k);

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
                            var t = -Internals.Dot(n - kp1, vtColk + kp1, vtColj + kp1) / vtColk[kp1];
                            Internals.MulAdd(n - k, vtColk + k, t, vtColj + k);
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
                        Internals.MulInplace(m, u + ldu * i, r);
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
                        Internals.MulInplace(n, vt + ldvt * (i + 1), r);
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
                if (iter >= 1000)
                {
                    throw new MatrixFactorizationException("SVD failed. The solution did not converge.");
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
                    if (Math.Abs(e[k]) <= eps * (Math.Abs(stmp[k]) + Math.Abs(stmp[k + 1])))
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
                    int ks;
                    for (ks = p - 1; ks > k; ks--)
                    {
                        var t = 0.0;
                        if (ks != p - 1)
                        {
                            t += Math.Abs(e[ks]);
                        }
                        if (ks != k + 1)
                        {
                            t += Math.Abs(e[ks - 1]);
                        }
                        if (Math.Abs(stmp[ks]) <= eps * t)
                        {
                            stmp[ks] = 0.0;
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
                        f = e[p - 2];
                        e[p - 2] = 0.0;
                        for (var j = k; j < p - 1; j++)
                        {
                            var l = p - 2 - j + k;
                            var t = stmp[l];
                            Internals.Drotg(ref t, ref f, out cs, out sn);
                            stmp[l] = t;
                            if (l != k)
                            {
                                f = -sn * e[l - 1];
                                e[l - 1] *= cs;
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
                        f = e[k - 1];
                        e[k - 1] = 0.0;
                        for (var j = k; j < p; j++)
                        {
                            var t = stmp[j];
                            Internals.Drotg(ref t, ref f, out cs, out sn);
                            stmp[j] = t;
                            f = -sn * e[j];
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
                        scale = Math.Max(scale, Math.Abs(stmp[p - 1]));
                        scale = Math.Max(scale, Math.Abs(stmp[p - 2]));
                        scale = Math.Max(scale, Math.Abs(e[p - 2]));
                        scale = Math.Max(scale, Math.Abs(stmp[k]));
                        scale = Math.Max(scale, Math.Abs(e[k]));
                        var sp = stmp[p - 1] / scale;
                        var spm1 = stmp[p - 2] / scale;
                        var epm1 = e[p - 2] / scale;
                        var sk = stmp[k] / scale;
                        var ek = e[k] / scale;
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
                            Internals.Drotg(ref f, ref g, out cs, out sn);
                            if (j != k)
                            {
                                e[j - 1] = f;
                            }

                            f = (cs * stmp[j]) + (sn * e[j]);
                            e[j] = (cs * e[j]) - (sn * stmp[j]);
                            g = sn * stmp[j + 1];
                            stmp[j + 1] *= cs;

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

                            Internals.Drotg(ref f, ref g, out cs, out sn);
                            stmp[j] = f;
                            f = (cs * e[j]) + (sn * stmp[j + 1]);
                            stmp[j + 1] = -(sn * e[j]) + (cs * stmp[j + 1]);
                            g = sn * e[j + 1];
                            e[j + 1] *= cs;

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
                        iter = iter++;

                        break;

                    // Convergence.
                    case 4:

                        // Make the singular value positive.
                        if (stmp[k] < 0.0)
                        {
                            stmp[k] = -stmp[k];

                            if (vt != null)
                            {
                                if (vt != null)
                                {
                                    Internals.FlipSign(n, vt + ldvt * k);
                                }
                            }
                        }

                        // Order the singular value.
                        while (k < pp)
                        {
                            if (stmp[k] >= stmp[k + 1])
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
                        p--;

                        break;
                }
            }

            if (vt != null)
            {
                var vtColj = vt;
                for (var j = 0; j < n; j++)
                {
                    var vtColi = vt;
                    for (var i = 0; i < j; i++)
                    {
                        (vtColj[i], vtColi[j]) = (vtColi[j], vtColj[i]);
                        vtColi += ldvt;
                    }
                    vtColj += ldvt;
                }
            }

            var copySize = sizeof(double) * Math.Min(m, n);
            Buffer.MemoryCopy(stmp, s, copySize, copySize);
        }

        private static unsafe void SvdCore(int m, int n, Complex* a, int lda, double* s, Complex* u, int ldu, Complex* vt, int ldvt, Complex* work, Complex* e, Complex* stmp)
        {
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
                    stmp[k] = Internals.Norm(m - k, aColk + k);

                    if (stmp[k] != Complex.Zero)
                    {
                        if (aColk[k] != Complex.Zero)
                        {
                            stmp[k] = Internals.ChangeArgument(stmp[k], aColk[k]);
                        }

                        Internals.DivInplace(m - k, aColk + k, stmp[k]);

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
                        var t = -Internals.DotConj(m - k, aColk + k, aColj + k) / aColk[k];
                        Internals.MulAdd(m - k, aColk + k, t, aColj + k);
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
                    e[k] = Internals.Norm(n - kp1, e + kp1);

                    if (e[k] != Complex.Zero)
                    {
                        if (e[kp1] != Complex.Zero)
                        {
                            e[k] = Internals.ChangeArgument(e[k], e[kp1]);
                        }

                        Internals.DivInplace(n - kp1, e + kp1, e[k]);

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
                            Internals.MulAdd(m - kp1, aColj + kp1, e[j], work + kp1);
                        }

                        for (var j = kp1; j < n; j++)
                        {
                            var aColj = a + lda * j;
                            Internals.MulAdd(m - kp1, work + kp1, (-e[j] / e[kp1]).Conjugate(), aColj + kp1);
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
                            var t = -Internals.DotConj(m - k, uColk + k, uColj + k) / uColk[k];
                            Internals.MulAdd(m - k, uColk + k, t, uColj + k);
                        }

                        Internals.FlipSign(m - k, uColk + k);

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
                            var t = -Internals.DotConj(n - kp1, vtColk + kp1, vtColj + kp1) / vtColk[kp1];
                            Internals.MulAdd(n - k, vtColk + k, t, vtColj + k);
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
                        Internals.MulInplace(m, u + ldu * i, r);
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
                        Internals.MulInplace(n, vt + ldvt * (i + 1), r);
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
                if (iter >= 1000)
                {
                    throw new MatrixFactorizationException("SVD failed. The solution did not converge.");
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
                            Internals.Drotg(ref t, ref f, out cs, out sn);
                            stmp[l] = t;
                            if (l != k)
                            {
                                f = -sn * e[l - 1].Real;
                                e[l - 1] *= cs;
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
                            Internals.Drotg(ref t, ref f, out cs, out sn);
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
                            Internals.Drotg(ref f, ref g, out cs, out sn);
                            if (j != k)
                            {
                                e[j - 1] = f;
                            }

                            f = cs * stmp[j].Real + sn * e[j].Real;
                            e[j] = cs * e[j] - sn * stmp[j];
                            g = sn * stmp[j + 1].Real;
                            stmp[j + 1] *= cs;

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

                            Internals.Drotg(ref f, ref g, out cs, out sn);
                            stmp[j] = f;
                            f = cs * e[j].Real + sn * stmp[j + 1].Real;
                            stmp[j + 1] = -(sn * e[j]) + (cs * stmp[j + 1]);
                            g = sn * e[j + 1].Real;
                            e[j + 1] *= cs;

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
                                Internals.FlipSign(n, vt + ldvt * k);
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
                        p--;

                        break;
                }
            }

            if (vt != null)
            {
                var vtColj = vt;
                for (var j = 0; j < n; j++)
                {
                    var vtColi = vt;
                    for (var i = 0; i <= j; i++)
                    {
                        if (i == j)
                        {
                            vtColj[j] = vtColj[j].Conjugate();
                        }
                        else
                        {
                            var t1 = vtColj[i];
                            var t2 = vtColi[j];
                            vtColj[i] = t2.Conjugate();
                            vtColi[j] = t1.Conjugate();
                        }
                        vtColi += ldvt;
                    }
                    vtColj += ldvt;
                }
            }

            var count = Math.Min(m, n);
            for (var i = 0; i < count; i++)
            {
                s[i] = stmp[i].Real;
            }
        }
    }
}
