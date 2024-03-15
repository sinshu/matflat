using System;
using System.Linq;
using System.Numerics;

namespace MatFlatTest
{
    public static class Matrix
    {
        public static float[] RandomSingle(int seed, int m, int n, int lda)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * lda).Select(i => random.NextSingle());
            return values.ToArray();
        }

        public static double[] RandomDouble(int seed, int m, int n, int lda)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * lda).Select(i => random.NextDouble());
            return values.ToArray();
        }

        public static Complex[] RandomComplex(int seed, int m, int n, int lda)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * lda).Select(i => new Complex(random.NextDouble(), random.NextDouble()));
            return values.ToArray();
        }

        public static double Get(int m, int n, double[] a, int lda, int row, int col)
        {
            var index = col * lda + row;
            return a[index];
        }

        public static double Set(int m, int n, double[] a, int lda, int row, int col, double value)
        {
            var index = col * lda + row;
            return a[index] = value;
        }

        public static void Print(int m, int n, double[] a, int lda)
        {
            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    Console.Write("\t");
                    Console.Write(Get(m, n, a, lda, row, col).ToString("G6"));
                }
                Console.WriteLine();
            }
        }
    }
}
