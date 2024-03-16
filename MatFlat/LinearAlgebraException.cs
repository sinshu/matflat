using System;
using System.Runtime.Serialization;

namespace MatFlat
{
    /// <summary>
    /// Represents errors that occur in linear algebra operations.
    /// </summary>
    public class LinearAlgebraException : Exception, ISerializable
    {
        /// <inheritdoc/>
        public LinearAlgebraException() : base()
        {
        }

        /// <inheritdoc/>
        public LinearAlgebraException(string message) : base(message)
        {
        }

        /// <inheritdoc/>
        public LinearAlgebraException(string message, Exception inner) : base(message, inner)
        {
        }

        /// <inheritdoc/>
        protected LinearAlgebraException(SerializationInfo info, StreamingContext context)
        {
        }
    }
}
