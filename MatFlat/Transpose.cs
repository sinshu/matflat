using System;

namespace MatFlat
{
    /// <summary>
    /// Specifies the matrix transposition.
    /// </summary>
    public enum Transpose : int
    {
        /// <summary>
        /// No transposition.
        /// </summary>
        NoTrans = 111,

        /// <summary>
        /// Transpose the matrix.
        /// </summary>
        Trans = 112,

        /// <summary>
        /// Conjugate and transpose the matrix.
        /// </summary>
        ConjTrans = 113,

        /// <summary>
        /// Conjugate the matrix without transposition.
        /// </summary>
        ConjNoTrans = 114,
    }
}
