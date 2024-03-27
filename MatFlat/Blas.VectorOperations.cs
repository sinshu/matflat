using System;
using System.Numerics;

namespace MatFlat
{
    public static partial class Blas
    {
        /// <summary>
        /// Computes the L2 norm of the vector x.
        /// </summary>
        /// <param name="n">
        /// The number of elements.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <returns>
        /// The L2 norm.
        /// </returns>
        public static unsafe float L2Norm(int n, float* x, int incx)
        {
            if (incx == 1)
            {
                return (float)Internals.Norm(n, x);
            }
            else
            {
                return (float)Internals.Norm(n, x, incx);
            }
        }

        /// <summary>
        /// Computes the L2 norm of the vector x.
        /// </summary>
        /// <param name="n">
        /// The number of elements.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <returns>
        /// The L2 norm.
        /// </returns>
        public static unsafe double L2Norm(int n, double* x, int incx)
        {
            if (incx == 1)
            {
                return Internals.Norm(n, x);
            }
            else
            {
                return Internals.Norm(n, x, incx);
            }
        }

        /// <summary>
        /// Computes the L2 norm of the vector x.
        /// </summary>
        /// <param name="n">
        /// The number of elements.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <returns>
        /// The L2 norm.
        /// </returns>
        public static unsafe double L2Norm(int n, Complex* x, int incx)
        {
            if (incx == 1)
            {
                return Internals.Norm(n, x);
            }
            else
            {
                return Internals.Norm(n, x, incx);
            }
        }

        /// <summary>
        /// Computes the dot product <c>x^T * y</c>.
        /// </summary>
        /// <param name="n">
        /// The number of elements.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        /// <returns>
        /// The dot product.
        /// </returns>
        public static unsafe float Dot(int n, float* x, int incx, float* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                return (float)Internals.Dot(n, x, y);
            }
            else
            {
                return (float)Internals.Dot(n, x, incx, y, incy);
            }
        }

        /// <summary>
        /// Computes the dot product <c>x^T * y</c>.
        /// </summary>
        /// <param name="n">
        /// The number of elements.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        /// <returns>
        /// The dot product.
        /// </returns>
        public static unsafe double Dot(int n, double* x, int incx, double* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                return Internals.Dot(n, x, y);
            }
            else
            {
                return Internals.Dot(n, x, incx, y, incy);
            }
        }

        /// <summary>
        /// Computes the dot product <c>x^T * y</c>.
        /// </summary>
        /// <param name="n">
        /// The number of elements.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        /// <returns>
        /// The dot product.
        /// </returns>
        public static unsafe Complex Dot(int n, Complex* x, int incx, Complex* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                return Internals.Dot(n, x, y);
            }
            else
            {
                return Internals.Dot(n, x, incx, y, incy);
            }
        }

        /// <summary>
        /// Computes the dot product <c>x^H * y</c>.
        /// </summary>
        /// <param name="n">
        /// The number of elements.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        /// <returns>
        /// The dot product.
        /// </returns>
        public static unsafe Complex DotConj(int n, Complex* x, int incx, Complex* y, int incy)
        {
            if (incx == 1 && incy == 1)
            {
                return Internals.DotConj(n, x, y);
            }
            else
            {
                return Internals.DotConj(n, x, incx, y, incy);
            }
        }

        /// <summary>
        /// Computes the outer product <c>A = x * y^T</c>.
        /// </summary>
        /// <param name="m">
        /// The number of rows of the matrix A.
        /// </param>
        /// <param name="n">
        /// The number of columns of the matrix A.
        /// </param>
        /// <param name="x">
        /// The vector x.
        /// </param>
        /// <param name="incx">
        /// The stride for the elements of the array x.
        /// </param>
        /// <param name="y">
        /// The vector y.
        /// </param>
        /// <param name="incy">
        /// The stride for the elements of the array y.
        /// </param>
        /// <param name="a">
        /// The matrix A, which is the result of the multiplication.
        /// </param>
        /// <param name="lda">
        /// The leading dimension of the array A.
        /// </param>
        public static unsafe void Outer(int m, int n, double* x, int incx, double* y, int incy, double* a, int lda)
        {
        }
    }
}
