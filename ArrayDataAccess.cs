using System.Collections;

namespace AdamOptimizer
{
    public class ArrayDataAccess<T> : IDataAccess<T>
    {
        public ArrayDataAccess(T[] array)
        {
            Array = array;
        }
        public ArrayDataAccess(int length)
        {
            Array = new T[length];
        }
        public T this[int index]
        {
            get => Array[index];
            set => Array[index] = value;
        }

        public T[] Array { get; }

        public int Length => Array.Length;
        public IEnumerator<T> GetEnumerator()
        {
            foreach (var v in Array) yield return v;
        }
        IEnumerator IEnumerable.GetEnumerator() => Array.GetEnumerator();
        public static implicit operator T[](ArrayDataAccess<T> t)=>t.Array;
        public static implicit operator ArrayDataAccess<T>(T[] t)=>new(t);
    }
}