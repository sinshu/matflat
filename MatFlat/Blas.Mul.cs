using System;
using System.Numerics;

namespace MatFlat
{
    public static partial class Blas
    {
        public static unsafe void MulMatMat(Transpose transa, Transpose transb, int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
        {
            if (transa == Transpose.NoTrans)
            {
                if (transb == Transpose.NoTrans)
                {
                    var bColj = b;
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = Internals.Dot(k, a + i, lda, bColj, 1);
                        }
                        bColj += ldb;
                        cColj += ldc;
                    }
                }
                else
                {
                    throw new Exception();
                }
            }
            else
            {
                if (transb == Transpose.NoTrans)
                {
                    var bColj = b;
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        var aColi = a;
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = Internals.Dot(k, aColi, bColj);
                            aColi += lda;
                        }
                        bColj += ldb;
                        cColj += ldc;
                    }
                }
                else
                {
                    throw new Exception();
                }
            }
        }
    }
}
