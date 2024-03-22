using System;

public static class Program
{
    public static unsafe void Main(string[] args)
    {
        var a = new double[]
        {
            1, 2, 3,
            0, 4, 9,
            0, 0, 5,
        };
        var x = new double[] { 3, 4, 2 };
        for (var row = 0; row < 3; row++)
        {
            for (var col = 0; col < 3; col++)
            {
                var index = col * 3 + row;
                Console.Write(a[index] + ", ");
            }
            Console.WriteLine();
        }

        fixed (double* pa = a)
        fixed (double* px = x)
        {
            MatFlat.Blas.ForwardSubstitution(3, pa, 3, px, 1);
        }

        foreach (var value in x)
        {
            Console.WriteLine(value);
        }

        x = new double[] { 3, 4, 2 };
    }
}
