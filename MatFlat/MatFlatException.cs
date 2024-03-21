using System;
using System.Runtime.Serialization;

namespace MatFlat
{
    /// <summary>
    /// Represents errors from MatFlat.
    /// </summary>
    public class MatFlatException : Exception, ISerializable
    {
        /// <inheritdoc/>
        public MatFlatException() : base()
        {
        }

        /// <inheritdoc/>
        public MatFlatException(string message) : base(message)
        {
        }

        /// <inheritdoc/>
        public MatFlatException(string message, Exception inner) : base(message, inner)
        {
        }

        /// <inheritdoc/>
        protected MatFlatException(SerializationInfo info, StreamingContext context)
        {
        }
    }
}
