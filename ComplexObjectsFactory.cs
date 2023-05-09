using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AdamOptimizer;

public class ComplexObjectsFactory
{
    public ComplexObjectsFactory(IDataAccess<double> data)
    {
        this.Data = data;
    }
    public IDataAccess<double> Data { get; }
    int CurrentPosition = 0;
    public IDataAccess<double> TakeDouble(int count){
        var res = new PartialDataAccess<double>(Data,CurrentPosition,count);
        CurrentPosition+=count;
        ThrowIfOutOfRange();
        return res;
    }
    public CustomVector TakeVector(int length){
        var res = new CustomVector(new CustomVectorStorage(Data,CurrentPosition,length));
        CurrentPosition+=length;
        ThrowIfOutOfRange();
        return res;
    }
    public CustomMatrix TakeMatrix(int rows, int columns){
        var res = new CustomMatrix(new CustomMatrixStorage(Data,CurrentPosition,rows,columns));
        CurrentPosition+=rows*columns;
        ThrowIfOutOfRange();
        return res;
    }
    private void ThrowIfOutOfRange()
    {
        if(CurrentPosition>=Data.Length)
            throw new Exception("In ComplexObjectsFactory you took too much objects, so much that there is no space left in underlying IDataAccess");
    }
}