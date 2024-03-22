using System;
using System.Linq;
using System.Numerics;

namespace MatFlatTest
{
    public static class Vector
    {
        public static float[] RandomSingle(int seed, int n, int inc)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * inc).Select(i => 2 * random.NextSingle() - 1);
            return values.ToArray();
        }

        public static double[] RandomDouble(int seed, int n, int inc)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * inc).Select(i => 2 * random.NextDouble() - 1);
            return values.ToArray();
        }

        public static Complex[] RandomComplex(int seed, int n, int inc)
        {
            var random = new Random(seed);
            var values = Enumerable.Range(0, n * inc).Select(i => new Complex(2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1));
            return values.ToArray();
        }
    }
}
