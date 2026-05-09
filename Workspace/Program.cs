using System;
using System.IO;
using MatFlat;

public static class Program
{
    public static unsafe void Main(string[] args)
    {
        // MatFlat stores matrices in column-major order.
        // This is the 2-by-2 matrix:
        //     [ 0  1 ]
        //     [ 0  2 ]
        // During the double SVD implementation's initial bidiagonal setup, it has
        // stmp = [0, 2] and e = [1, 0]. Since e[0] is not negligible but stmp[0]
        // is negligible, the first main-iteration dispatch is kase = 2.
        const int m = 2;
        const int n = 2;
        var a = new double[]
        {
            0.0, 0.0,
            1.0, 2.0,
        };

        var initialKase = ComputeInitialSvdKaseForThisExample();
        Console.WriteLine("Minimal double SVD kase2 example");
        Console.WriteLine("A = [ [0, 1], [0, 2] ]");
        Console.WriteLine("column-major storage = [0, 0, 1, 2]");
        Console.WriteLine("initial bidiagonal stmp = [0, 2], e = [1, 0]");
        Console.WriteLine($"first main-iteration kase = {initialKase}");
        Console.WriteLine();

        var s = new double[Math.Min(m, n)];
        var u = new double[m * m];
        var vt = new double[n * n];

        var trace = RunSvdAndCaptureKaseTrace(m, n, a, s, u, vt);
        Console.Write(trace);

        if (!trace.Contains("MatFlat.Svd<double>: reached case 2", StringComparison.Ordinal))
        {
            throw new InvalidOperationException("The actual Factorization.Svd execution did not reach double SVD case 2.");
        }

        Console.WriteLine("Confirmed: actual Factorization.Svd execution reached double SVD case 2.");
        Console.WriteLine();

        Console.WriteLine("SVD completed. Singular values:");
        foreach (var value in s)
        {
            Console.WriteLine(value);
        }
    }

    private static unsafe string RunSvdAndCaptureKaseTrace(int m, int n, double[] a, double[] s, double[] u, double[] vt)
    {
        using var writer = new StringWriter();
        var originalOut = Console.Out;
        var originalTraceSetting = Environment.GetEnvironmentVariable("MATFLAT_TRACE_SVD_KASE");

        try
        {
            Environment.SetEnvironmentVariable("MATFLAT_TRACE_SVD_KASE", "1");
            Console.SetOut(writer);

            fixed (double* pa = a)
            fixed (double* ps = s)
            fixed (double* pu = u)
            fixed (double* pvt = vt)
            {
                Factorization.Svd(m, n, pa, m, ps, pu, m, pvt, n);
            }
        }
        finally
        {
            Console.SetOut(originalOut);
            Environment.SetEnvironmentVariable("MATFLAT_TRACE_SVD_KASE", originalTraceSetting);
        }

        return writer.ToString();
    }

    private static int ComputeInitialSvdKaseForThisExample()
    {
        var stmp = new[] { 0.0, 2.0 };
        var e = new[] { 1.0, 0.0 };
        var p = 2;
        var eps = Math.Pow(2.0, -52.0);

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
            }
        }

        return kase;
    }
}
