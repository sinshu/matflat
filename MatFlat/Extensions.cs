using System;
using System.Numerics;

namespace MatFlat
{
    internal static class Extensions
    {
        internal static Complex Conjugate(this Complex value)
        {
            return new Complex(value.Real, -value.Imaginary);
        }
    }
}
