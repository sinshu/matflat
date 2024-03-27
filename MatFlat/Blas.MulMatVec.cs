using System;
using System.Numerics;

namespace MatFlat
{
    public static partial class Blas
    {
        /// <summary>
        /// Computes a matrix-and-vector multiplication <c>y = A * x</c>.
        /// </summary>
        /// <param name="trans">
        /// Specifies whether the matrix A is treated as transposed or not.
        /// </param>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// The matrix A.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y, which is the result of the multiplication.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        public static unsafe void MulMatVec(Transpose trans, int m, int n, float* a, int lda, float* x, int incx, float* y, int incy)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (x == null)
            {
                throw new ArgumentNullException(nameof(x));
            }

            if (incx <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(incx));
            }

            if (y == null)
            {
                throw new ArgumentNullException(nameof(y));
            }

            if (incy <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(incy));
            }

            if (a == y || x == y)
            {
                throw new ArgumentException("In-place operation is not supported.");
            }

            if (trans == Transpose.NoTrans)
            {
                for (var i = 0; i < m; i++)
                {
                    *y = (float)Internals.Dot(n, a + i, lda, x, incx);
                    y += incy;
                }
            }
            else if (trans == Transpose.Trans)
            {
                for (var i = 0; i < n; i++)
                {
                    *y = (float)Internals.Dot(m, a, 1, x, incx);
                    a += lda;
                    y += incy;
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(trans));
            }
        }

        /// <summary>
        /// Computes a matrix-and-vector multiplication <c>y = A * x</c>.
        /// </summary>
        /// <param name="trans">
        /// Specifies whether the matrix A is treated as transposed or not.
        /// </param>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// The matrix A.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y, which is the result of the multiplication.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        public static unsafe void MulMatVec(Transpose trans, int m, int n, double* a, int lda, double* x, int incx, double* y, int incy)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (x == null)
            {
                throw new ArgumentNullException(nameof(x));
            }

            if (incx <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(incx));
            }

            if (y == null)
            {
                throw new ArgumentNullException(nameof(y));
            }

            if (incy <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(incy));
            }

            if (a == y || x == y)
            {
                throw new ArgumentException("In-place operation is not supported.");
            }

            if (trans == Transpose.NoTrans)
            {
                for (var i = 0; i < m; i++)
                {
                    *y = Internals.Dot(n, a + i, lda, x, incx);
                    y += incy;
                }
            }
            else if (trans == Transpose.Trans)
            {
                for (var i = 0; i < n; i++)
                {
                    *y = Internals.Dot(m, a, 1, x, incx);
                    a += lda;
                    y += incy;
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(trans));
            }
        }

        /// <summary>
        /// Computes a matrix-and-vector multiplication <c>y = A * x</c>.
        /// </summary>
        /// <param name="trans">
        /// Specifies whether the matrix A is treated as transposed or not.
        /// </param>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="a">
        /// The matrix A.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y, which is the result of the multiplication.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        public static unsafe void MulMatVec(Transpose trans, int m, int n, Complex* a, int lda, Complex* x, int incx, Complex* y, int incy)
        {
            if (m <= 0)
            {
                throw new ArgumentException("The number of rows must be greater than or equal to one.", nameof(m));
            }

            if (n <= 0)
            {
                throw new ArgumentException("The number of columns must be greater than or equal to one.", nameof(n));
            }

            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (lda < m)
            {
                throw new ArgumentException("The leading dimension must be greater than or equal to the number of rows.", nameof(lda));
            }

            if (x == null)
            {
                throw new ArgumentNullException(nameof(x));
            }

            if (incx <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(incx));
            }

            if (y == null)
            {
                throw new ArgumentNullException(nameof(y));
            }

            if (incy <= 0)
            {
                throw new ArgumentException("The value must be greater than or equal to one.", nameof(incy));
            }

            if (a == y || x == y)
            {
                throw new ArgumentException("In-place operation is not supported.");
            }

            if (trans == Transpose.NoTrans)
            {
                for (var i = 0; i < m; i++)
                {
                    *y = Internals.Dot(n, a + i, lda, x, incx);
                    y += incy;
                }
            }
            else if (trans == Transpose.Trans)
            {
                for (var i = 0; i < n; i++)
                {
                    *y = Internals.Dot(m, a, 1, x, incx);
                    a += lda;
                    y += incy;
                }
            }
            else
            {
                throw new ArgumentException("Invalid enum value.", nameof(trans));
            }
        }
    }
}
