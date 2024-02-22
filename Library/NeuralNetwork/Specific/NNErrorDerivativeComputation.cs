namespace GradientDescentSharp.NeuralNetwork.Specific;

internal class NNErrorDerivativeComputation : NNBase{
    /// <summary>
    /// See the class description to understand what it does
    /// </summary>
    LossFunctionOutputDerivativeReplacer replacer;
    public NNErrorDerivativeComputation(NNBase baseModel, LossFunctionOutputDerivativeReplacer replacer) : base(baseModel){
        this.replacer = replacer;
    }
    public override FTensor Forward(FTensor input){
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Activation.Activation(layer.Forward(input));
        }
        replacer.ReplaceOutputParameter(input);
        return input;
    }
}
