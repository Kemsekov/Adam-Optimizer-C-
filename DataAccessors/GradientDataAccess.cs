using System.Collections;

namespace AdamOptimizer
{
    /// <summary>
    /// When computing gradients we need to change some value of variables by epsilon <br/>
    /// and recompute function on it.  <br/>
    /// Because re-creating a new data variables is  <br/>
    /// problematic and consuming, this class allows to create a new IDataAccess object <br/>
    /// that does only one thing: on some index instead of original value it <br/>
    /// returns changed value. <br/>
    /// By doing this we avoid all problems with memory and it also allows us to
    /// compute gradients in parallel!
    /// </summary>
    public class GradientDataAccess<T> : IDataAccess<T>{
        public GradientDataAccess(IDataAccess<T> original, int changedIndex, T changedValue)
        {
            DataAccess=original;
            ChangedIndex = changedIndex;
            ChangedValue = changedValue;
        }

        public T this[int index] { 
            get => (index==ChangedIndex) ? ChangedValue : DataAccess[index]; 
            set{
                DataAccess[index]=value;
                ChangedValue=value;
            }
        }
        public void Reset(int changedIndex, T changedValue){
            ChangedIndex = changedIndex;
            ChangedValue = changedValue;
        }
        public IDataAccess<T> DataAccess { get; }
        public int ChangedIndex { get;protected set; }
        public T ChangedValue { get; protected set;}
        public int Length => DataAccess.Length;

        public IEnumerator<T> GetEnumerator()
        {
            for(int i = 0;i<Length;i++){
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}