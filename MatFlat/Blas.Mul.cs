using System;
using System.Numerics;

namespace MatFlat
{
    public static partial class Blas
    {
        /// <summary>
        /// Computes a matrix multiplication <c>C = A * B</c>.
        /// </summary>
        /// <param name="transa">
        /// Specifies whether the matrix A is treated as transposed or not.
        /// </param>
        /// <param name="transb">
        /// Specifies whether the matrix B is treated as transposed or not.
        /// </param>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix B.
        /// </param>
        /// <param name="k">
        /// The number of columns of the matrix A.
        /// This is equal to the number of rows of the matrix B.
        /// </param>
        /// <param name="a">
        /// The matrix A.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="b">
        /// The matrix B.
        /// </param>
        /// <param name="ldb">
        /// The leading dimension of the array B.
        /// </param>
        /// <param name="c">
        /// The matrix C, which is the result of the multiplication.
        /// </param>
        /// <param name="ldc">
        /// The leading dimension of the array C.
        /// </param>
        public static unsafe void MulMatMat(Transpose transa, Transpose transb, int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(n));
            }

            if (k <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(k));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (transa == Transpose.NoTrans)
            {
                if (lda < m)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
                }
            }
            else if (transa == Transpose.Trans)
            {
                if (lda < k)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(transa));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (transb == Transpose.NoTrans)
            {
                if (ldb < k)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldb));
                }
            }
            else if (transb == Transpose.Trans)
            {
                if (ldb < n)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldb));
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(transb));
            }

            if (c == null)
            {
                throw new ArgumentNullException(nameof(c));
            }

            if (ldc < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldc));
            }

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
                            cColj[i] = (float)Internals.Dot(k, a + i, lda, bColj, 1);
                        }
                        bColj += ldb;
                        cColj += ldc;
                    }
                }
                else
                {
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = (float)Internals.Dot(k, a + i, lda, b + j, ldb);
                        }
                        cColj += ldc;
                    }
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
                            cColj[i] = (float)Internals.Dot(k, aColi, bColj);
                            aColi += lda;
                        }
                        bColj += ldb;
                        cColj += ldc;
                    }
                }
                else
                {
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        var aColi = a;
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = (float)Internals.Dot(k, aColi, 1, b + j, ldb);
                            aColi += lda;
                        }
                        cColj += ldc;
                    }
                }
            }
        }

        /// <summary>
        /// Computes a matrix multiplication <c>C = A * B</c>.
        /// </summary>
        /// <param name="transa">
        /// Specifies whether the matrix A is treated as transposed or not.
        /// </param>
        /// <param name="transb">
        /// Specifies whether the matrix B is treated as transposed or not.
        /// </param>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix B.
        /// </param>
        /// <param name="k">
        /// The number of columns of the matrix A.
        /// This is equal to the number of rows of the matrix B.
        /// </param>
        /// <param name="a">
        /// The matrix A.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="b">
        /// The matrix B.
        /// </param>
        /// <param name="ldb">
        /// The leading dimension of the array B.
        /// </param>
        /// <param name="c">
        /// The matrix C, which is the result of the multiplication.
        /// </param>
        /// <param name="ldc">
        /// The leading dimension of the array C.
        /// </param>
        public static unsafe void MulMatMat(Transpose transa, Transpose transb, int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(n));
            }

            if (k <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(k));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (transa == Transpose.NoTrans)
            {
                if (lda < m)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
                }
            }
            else if (transa == Transpose.Trans)
            {
                if (lda < k)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(transa));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (transb == Transpose.NoTrans)
            {
                if (ldb < k)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldb));
                }
            }
            else if (transb == Transpose.Trans)
            {
                if (ldb < n)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldb));
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(transb));
            }

            if (c == null)
            {
                throw new ArgumentNullException(nameof(c));
            }

            if (ldc < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldc));
            }

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
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = Internals.Dot(k, a + i, lda, b + j, ldb);
                        }
                        cColj += ldc;
                    }
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
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        var aColi = a;
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = Internals.Dot(k, aColi, 1, b + j, ldb);
                            aColi += lda;
                        }
                        cColj += ldc;
                    }
                }
            }
        }

        /// <summary>
        /// Computes a matrix multiplication <c>C = A * B</c>.
        /// </summary>
        /// <param name="transa">
        /// Specifies whether the matrix A is treated as transposed or not.
        /// </param>
        /// <param name="transb">
        /// Specifies whether the matrix B is treated as transposed or not.
        /// </param>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix B.
        /// </param>
        /// <param name="k">
        /// The number of columns of the matrix A.
        /// This is equal to the number of rows of the matrix B.
        /// </param>
        /// <param name="a">
        /// The matrix A.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="b">
        /// The matrix B.
        /// </param>
        /// <param name="ldb">
        /// The leading dimension of the array B.
        /// </param>
        /// <param name="c">
        /// The matrix C, which is the result of the multiplication.
        /// </param>
        /// <param name="ldc">
        /// The leading dimension of the array C.
        /// </param>
        public static unsafe void MulMatMat(Transpose transa, Transpose transb, int m, int n, int k, Complex* a, int lda, Complex* b, int ldb, Complex* c, int ldc)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(n));
            }

            if (k <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(k));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (transa == Transpose.NoTrans || transa == Transpose.ConjNoTrans)
            {
                if (lda < m)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
                }
            }
            else if (transa == Transpose.Trans || transa == Transpose.ConjTrans)
            {
                if (lda < k)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(transa));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (transb == Transpose.NoTrans || transb == Transpose.ConjNoTrans)
            {
                if (ldb < k)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldb));
                }
            }
            else if (transb == Transpose.Trans || transb == Transpose.ConjTrans)
            {
                if (ldb < n)
                {
                    throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldb));
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(transb));
            }

            if (c == null)
            {
                throw new ArgumentNullException(nameof(c));
            }

            if (ldc < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(ldc));
            }

            delegate*<int, Complex*, int, Complex*, int, Complex> dot;
            if (HasConj(transa))
            {
                if (HasConj(transb))
                {
                    dot = &Dot_CC;
                }
                else
                {
                    dot = &Dot_CN;
                }
            }
            else
            {
                if (HasConj(transb))
                {
                    dot = &Dot_NC;
                }
                else
                {
                    dot = &Dot_NN;
                }
            }
            transa = RemoveConj(transa);
            transb = RemoveConj(transb);

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
                            cColj[i] = dot(k, a + i, lda, bColj, 1);
                        }
                        bColj += ldb;
                        cColj += ldc;
                    }
                }
                else
                {
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = dot(k, a + i, lda, b + j, ldb);
                        }
                        cColj += ldc;
                    }
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
                            cColj[i] = dot(k, aColi, 1, bColj, 1);
                            aColi += lda;
                        }
                        bColj += ldb;
                        cColj += ldc;
                    }
                }
                else
                {
                    var cColj = c;
                    for (var j = 0; j < n; j++)
                    {
                        var aColi = a;
                        for (var i = 0; i < m; i++)
                        {
                            cColj[i] = dot(k, aColi, 1, b + j, ldb);
                            aColi += lda;
                        }
                        cColj += ldc;
                    }
                }
            }
        }

        private static bool HasConj(Transpose trans)
        {
            return trans == Transpose.ConjNoTrans || trans == Transpose.ConjTrans;
        }

        private static Transpose RemoveConj(Transpose trans)
        {
            if (trans == Transpose.ConjNoTrans)
            {
                return Transpose.NoTrans;
            }
            else if (trans == Transpose.ConjTrans)
            {
                return Transpose.Trans;
            }
            else
            {
                return trans;
            }
        }

        private static unsafe Complex Dot_NN(int n, Complex* x, int incx, Complex* y, int incy)
        {
            var sum = Complex.Zero;
            while (n > 0)
            {
                sum += *x * *y;
                x += incx;
                y += incy;
                n--;
            }
            return sum;
        }

        private static unsafe Complex Dot_CN(int n, Complex* x, int incx, Complex* y, int incy)
        {
            var sum = Complex.Zero;
            while (n > 0)
            {
                sum += Internals.MulConj(*x, *y);
                x += incx;
                y += incy;
                n--;
            }
            return sum;
        }

        private static unsafe Complex Dot_NC(int n, Complex* x, int incx, Complex* y, int incy)
        {
            var sum = Complex.Zero;
            while (n > 0)
            {
                sum += Internals.MulConj(*y, *x);
                x += incx;
                y += incy;
                n--;
            }
            return sum;
        }

        private static unsafe Complex Dot_CC(int n, Complex* x, int incx, Complex* y, int incy)
        {
            var sum = Complex.Zero;
            while (n > 0)
            {
                sum += Internals.MulConjConj(*x, *y);
                x += incx;
                y += incy;
                n--;
            }
            return sum;
        }
    }
}
