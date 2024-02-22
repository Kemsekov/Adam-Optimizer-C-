using GradientDescentSharp.NeuralNetwork.Specific;
using MathNet.Numerics.LinearAlgebra.Single;
using Tensornet;

namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Extensions
/// </summary>
public static class NNBaseExtensions{

    ///<inheritdoc/>
    public static FTensor ToTensor(this FVector input){
        var inputT = Tensor.Zeros<float>(new(input.Count,1));
        inputT.VecMapInplace((i,v)=>input[i]);
        return inputT;
    }
    ///<inheritdoc/>
    public static FTensor ToTensor(this float[] input){
        var inputT = Tensor.Zeros<float>(new(input.Length,1));
        inputT.VecMapInplace((i,v)=>input[i]);
        return inputT;
    }
    ///<inheritdoc/>
    public static FVector ToFVector(this FTensor tensor){
        var span = tensor.AsSpan();
        var vec  = DenseVector.OfArray(span.ToArray());
        return vec;
    }
    ///<inheritdoc/>
    public static float[] ToArray(FTensor tensor){
        var span = tensor.AsSpan();
        return span.ToArray();
    }
    //--------------
    ///<inheritdoc cref="PredictOnlyNN.Error(Tensor{float}, Tensor{float})"/>
    public static float Error(this PredictOnlyNN nn,FVector input, FVector expected){
        return nn.Error(ToTensor(input),ToTensor(expected));
    }
    ///<inheritdoc cref="PredictOnlyNN.Forward(Tensor{float})"/>
    public static FVector Forward(this PredictOnlyNN nn, FVector input){
        return ToFVector(nn.Forward(ToTensor(input)));
    }
    //--------------
    ///<inheritdoc cref="NNBase.Error(Tensor{float}, Tensor{float})"/>
    public static float Error<T>(this T nn,FVector input, FVector expected)
    where T : NNBase
    {
        return nn.Error(ToTensor(input),ToTensor(expected));
    }
    ///<inheritdoc cref="NNBase.Error(Tensor{float}, Tensor{float})"/>
    public static float Error<T>(this T nn,float[] input, float[] expected)
    where T : NNBase
    {
        return nn.Error(ToTensor(input),ToTensor(expected));
    }
    ///<inheritdoc cref="NNBase.Forward(Tensor{float})"/>
    public static FVector Forward<T>(this T nn, FVector input)
    where T : NNBase
    {
        return ToFVector(nn.Forward(ToTensor(input)));
    }
     ///<inheritdoc cref="NNBase.Forward(Tensor{float})"/>
    public static float[] Forward<T>(this T nn, float[] input)
    where T : NNBase
    {
        return ToArray(nn.Forward(ToTensor(input)));
    }
    //--------------
     ///<inheritdoc cref="NNBase.LearnOnLoss"/>
    public static BackpropResult LearnOnLoss<T>(this T nn,FVector input, float theta, Func<FVector, PredictOnlyNN, float> lossFunction)
    where T : NNBase
    {
        return nn.LearnOnLoss(ToTensor(input),theta,(v,n)=>lossFunction(ToFVector(v),n));
    }
    ///<inheritdoc cref="NNBase.LearnOnLoss"/>
    public static BackpropResult LearnOnLoss<T>(this T nn,float[] input, float theta, Func<float[], PredictOnlyNN, float> lossFunction)
    where T : NNBase
    {
        return nn.LearnOnLoss(ToTensor(input),theta,(v,n)=>lossFunction(ToArray(v),n));
    }
    //--------------
    ///<inheritdoc cref="NNBase.Backwards"/>
    public static BackpropResult Backwards<T>(this T nn,FVector input, FVector expected)
    where T : NNBase
    {
        return nn.Backwards(ToTensor(input),ToTensor(expected));
    }
    ///<inheritdoc cref="NNBase.Backwards"/>
    public static BackpropResult Backwards<T>(this T nn,float[] input, float[] expected)
    where T : NNBase
    {
        return nn.Backwards(ToTensor(input),ToTensor(expected));
    }

}