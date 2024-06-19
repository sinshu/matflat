using System;
using MatFlat;

public static class Program
{
    public static unsafe void Main(string[] args)
    {
        var n = 3;
        var a = new double[n * n];

        for (var i = 0; i < a.Length; i++)
        {
            a[i] = double.NaN;
        }

        var u = new double[n * n];
        var s = new double[n];
        fixed (double* pa = a)
        fixed (double* pu = u)
        fixed (double* ps = s)
        {
            Factorization.Svd(n, n, pa, n, ps, pu, n, null, 0);
        }
        foreach (var value in u)
        {
            Console.WriteLine(value);
        }
    }
}
