using System;
using System.Linq;
using System.Numerics;
using NUnit.Framework;
using OpenBlasSharp;
using MatFlat;

namespace MatFlatTest
{
    public class CholeskyTests
    {
        [TestCase(3, 3)]
        public unsafe void CholeskyDouble_General(int n, int lda)
        {
            var a = Matrix.RandomDouble(42, n, n, lda);
            for (var row = 0; row < n; row++)
            {
                for (var col = 0; col <= row; col++)
                {
                    var index = lda * col + row;
                    if (row == col)
                    {
                        a[index] += 1.0;
                    }
                    else
                    {
                        a[index] = a[lda * row + col];
                    }
                }
            }

            var actualA = a.ToArray();

            Matrix.Print(n, n, actualA, lda);
            Console.WriteLine();

            fixed (double* pa = actualA)
            {
                Factorization.CholeskyDouble(n, pa, lda);
            }

            Matrix.Print(n, n, actualA, lda);
        }
    }
}
