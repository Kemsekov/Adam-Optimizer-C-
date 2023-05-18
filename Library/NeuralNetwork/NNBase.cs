using MathNet.Numerics.LinearAlgebra;

namespace GradientDescentSharp.NeuralNetwork;
public class BackpropResult
{
    private ILayer[] layers;
    public BackpropResult(ILayer[] layers)
    {
        this.layers = layers;
    }
    public void Unlearn()
    {
        foreach (var l in layers)
            l.Unlearn();
    }
}
public abstract class NNBase
{
    public ILayer[] Layers { get; }
    public FloatType LearningRate = 0.05;
    public NNBase(params ILayer[] layers)
    {
        Layers = layers;
        RawLayerOutput = new Dictionary<ILayer, Vector>();
        foreach(var layer in layers){
            RawLayerOutput[layer] = (Vector)layer.Bias.Map(x=>0.0);
        }
    }
    public Vector Forward(Vector input)
    {
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            input.MapInplace(x=>layer.Activation.Activation(x));
        }
        return input;
    }
    Dictionary<ILayer,Vector> RawLayerOutput;
    Vector ForwardForLearning(Vector input)
    {
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            this.RawLayerOutput[layer].MapIndexedInplace((index,x)=>input[index]);
            input.MapInplace(x=>layer.Activation.Activation(x));
        }
        return input;
    }
    public BackpropResult LearnOnError(Vector input,double theta, Func<Vector,NNBase,double> errorFunction){
        var original = errorFunction(input,this);
        var errorDerivative = input.Map(x=>0.0);
        for(int i = 0;i<input.Count;i++){
            input[i]+=theta;
            var changed = errorFunction(input,this);
            input[i]-=theta;
            errorDerivative[i]=(changed-original)/theta;
        }
        //fill layers with learning info
        ForwardForLearning(input);
        Learn(input,errorDerivative);
        return new BackpropResult(Layers);
    }

    public double Error(Vector input, Vector expected)
    {
        return (Forward(input) - expected).Sum(x => x * x);
    }
    
    void Learn(Vector input,Vector<double> errorDerivative)
    {
        for (int i = Layers.Length - 1; i >= 0; i--)
        {
            var layer = Layers[i];

            var totalWeightsSum = layer.Bias.Sum(b => b * b) + layer.Weights.ToEnumerable().Sum(x => x * x);
            var learningRate = LearningRate / totalWeightsSum;

            var biases = layer.Bias;
            var layerOutput = RawLayerOutput[layer];
            var activation = layer.Activation.Activation;
            var derivative = layer.Activation.ActivationDerivative;

            var biasesGradient = layerOutput.MapIndexed((j, x) => derivative(x) * errorDerivative[j]);
            if (i > 0)
            {
                errorDerivative.MapIndexedInplace((j, x) => x * derivative(layerOutput[j]));
                errorDerivative *= layer.Weights;
            }
            var layerInput = i > 0 ? RawLayerOutput[Layers[i - 1]].Map(activation) : input;

            layer.Learn((Vector)biasesGradient, (Vector)layerInput, learningRate);
        }
    }
    /// <summary>
    /// Modified backpropogation implementation, that inspired from MineDescent implementation.<br/>
    /// When backpropagation damages performance more than improves it, it rollback all changes to weights.<br/>
    /// Also learning rate for each layer depends on total weights sum, so it's step is normalized,
    /// which allows to learn a lot faster
    /// </summary>
    /// <param name="input"></param>
    /// <param name="expected"></param>
    /// <returns>True if learning was successful, False if backprop was unsuccessful and rolled back.</returns>
    public BackpropResult Backwards(Vector input, Vector expected)
    {
        // compute error derivative for MSE
        var error = (ForwardForLearning(input) - expected) * 2;
        Learn(input,error);
        return new BackpropResult(Layers);
    }
}