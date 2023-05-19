using MathNet.Numerics.LinearAlgebra;

namespace GradientDescentSharp.NeuralNetwork;

public abstract class NNBase
{
    public ILayer[] Layers { get; }
    public double LearningRate = 0.05;
    Dictionary<ILayer,Vector> RawLayerOutput;
    /// <summary>
    /// See the class description to understand what it does
    /// </summary>
    ErrorFunctionOutputDerivativeReplacer replacer;
    public NNBase(params ILayer[] layers)
    {
        replacer = new();
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
        replacer.ReplaceOutputParameter(input);
        return input;
    }

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
        replacer.ReplaceOutputParameter(input);
        return input;
    }

    public BackpropResult LearnOnError(Vector input,double theta, Func<Vector,NNBase,double> errorFunction){
        var original = errorFunction(input,this);
        var originalOutput = Forward(input);
        var errorDerivative = originalOutput.Map(x=>0.0);
        for(int i = 0;i<originalOutput.Count;i++){
            replacer.ChangedOutputIndex = i;
            replacer.ChangedOutputTheta=theta;
            var changed = errorFunction(input,this);
            errorDerivative[i]=((changed-original)/theta);
            replacer.ChangedOutputIndex = -1;
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