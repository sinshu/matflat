using System;

namespace MatFlat
{
    public static partial class MatrixDecomposition
    {
        public static void LuDouble(int m, int n, Span<double> a, int lda, Span<int> piv)
        {
            // Initialize the pivot matrix to the identity permutation.
            for (var i = 0; i < piv.Length; i++)
            {
                piv[i] = i;
            }

            var vecLUcolj = new double[m];

            // Outer loop.
            for (var j = 0; j < n; j++)
            {
                var indexj = j * lda;
                var indexjj = indexj + j;

                // Make a copy of the j-th column to localize references.
                for (var i = 0; i < vecLUcolj.Length; i++)
                {
                    vecLUcolj[i] = a[indexj + i];
                }

                // Apply previous transformations.
                for (var i = 0; i < m; i++)
                {
                    // Most of the time is spent in the following dot product.
                    var kmax = Math.Min(i, j);
                    var s = 0.0;
                    for (var k = 0; k < kmax; k++)
                    {
                        s += a[(k * lda) + i] * vecLUcolj[k];
                    }

                    a[indexj + i] = vecLUcolj[i] -= s;
                }

                // Find pivot and exchange if necessary.
                var p = j;
                for (var i = j + 1; i < m; i++)
                {
                    if (Math.Abs(vecLUcolj[i]) > Math.Abs(vecLUcolj[p]))
                    {
                        p = i;
                    }
                }

                if (p != j)
                {
                    for (var k = 0; k < n; k++)
                    {
                        var indexk = k * lda;
                        var indexkp = indexk + p;
                        var indexkj = indexk + j;
                        (a[indexkp], a[indexkj]) = (a[indexkj], a[indexkp]);
                    }

                    piv[j] = p;
                }

                // Compute multipliers.
                if (j < m && a[indexjj] != 0.0)
                {
                    for (var i = j + 1; i < m; i++)
                    {
                        a[indexj + i] /= a[indexjj];
                    }
                }
            }
        }
    }
}
