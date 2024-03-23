using System;

namespace MatFlat
{
    /// <summary>
    /// Specifies whether the upper or lower triangular part of the array is to be referenced.
    /// </summary>
    public enum Uplo : int
    {
        /// <summary>
        /// Only the upper triangular part of the array is to be referenced.
        /// </summary>
        Upper = 121,

        /// <summary>
        /// Only the lower triangular part of the array is to be referenced.
        /// </summary>
        Lower = 122,
    }
}
