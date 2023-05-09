using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AdamOptimizer
{
    public interface IDataAccess<T> : IEnumerable<T>
    {
        public int Length { get; }
        public T this[int index] { get; set; }
    }
}