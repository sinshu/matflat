﻿using System;
using System.Linq;
using System.Numerics;

namespace MatFlatTest
{
    public static class Matrix
    {
        public static float[] RandomSingle(int seed, int m, int n, int lda)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * lda).Select(i => 2 * random.NextSingle() - 1);
            return values.ToArray();
        }

        public static double[] RandomDouble(int seed, int m, int n, int lda)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * lda).Select(i => 2 * random.NextDouble() - 1);
            return values.ToArray();
        }

        public static Complex[] RandomComplex(int seed, int m, int n, int lda)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * lda).Select(i => new Complex(2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1));
            return values.ToArray();
        }

        public static T Get<T>(int m, int n, T[] a, int lda, int row, int col)
        {
            var index = col * lda + row;
            return a[index];
        }

        public static T Set<T>(int m, int n, T[] a, int lda, int row, int col, T value)
        {
            var index = col * lda + row;
            return a[index] = value;
        }

        public static void Print<T>(int m, int n, T[] a, int lda) where T : IFormattable
        {
            for (var row = 0; row < m; row++)
            {
                for (var col = 0; col < n; col++)
                {
                    Console.Write("\t");
                    Console.Write(Get(m, n, a, lda, row, col).ToString("G6", null));
                }
                Console.WriteLine();
            }
        }
    }
}
