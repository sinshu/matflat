using System;

public static class Program
{
    public static void Main(string[] args)
    {
        var x = new double[]
        {
            1, 2, 3,
            1, 4, 9,
            2, 2, 5,
        };

        PrintMatrix<double>(3, 3, x, 3);

        Console.WriteLine();

        MatFlat.MatrixDecomposition.LuDouble(3, 3, x, 3, new int[3]);

        PrintMatrix<double>(3, 3, x, 3);
    }

    private static void PrintMatrix<T>(int m, int n, Span<T> a, int lda)
    {
        for (var row = 0; row < m; row++)
        {
            for (var col = 0; col < n; col++)
            {
                var index = lda * col + row;
                Console.Write("\t");
                Console.Write(a[index]);
                Console.Write(",");
            }
            Console.WriteLine();
        }
    }
}
