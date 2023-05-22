namespace GradientDescentSharp.NeuralNetwork.Specific;
using MathNet.Numerics.LinearAlgebra.Single;

public class NNErrorDerivativeComputation : NNBase{
    /// <summary>
    /// See the class description to understand what it does
    /// </summary>
    LossFunctionOutputDerivativeReplacer replacer;
    public NNErrorDerivativeComputation(NNBase baseModel, LossFunctionOutputDerivativeReplacer replacer) : base(baseModel){
        this.replacer = replacer;
    }
    public override Vector Forward(Vector input){
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            input.MapInplace(x=>layer.Activation.Activation(x));
        }
        replacer.ReplaceOutputParameter(input);
        return input;
    }
    protected override Vector ForwardForLearning(Vector input){
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            RawLayerOutput[layer].MapIndexedInplace((index,x)=>input[index]);
            input.MapInplace(x=>layer.Activation.Activation(x));
        }
        replacer.ReplaceOutputParameter(input);
        return input;
    }
}
