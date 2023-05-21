using GradientDescentSharp.NeuralNetwork.Specific;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public abstract class NNBase
{
    public ILayer[] Layers { get; }
    public float LearningRate = 0.05f;
    protected Dictionary<ILayer,Vector> RawLayerOutput;

    /// <summary>
    /// Builds a copy of another neural network
    /// </summary>
    /// <param name="another"></param>
    public NNBase(NNBase another){
        Layers = another.Layers;
        RawLayerOutput = another.RawLayerOutput;
    }
    public NNBase(params ILayer[] layers)
    {
        Layers = layers;
        RawLayerOutput = new Dictionary<ILayer, Vector>();
        foreach(var layer in layers){
            RawLayerOutput[layer] = (Vector)layer.Bias.Map(x=>0.0f);
        }
    }
    public virtual Vector Forward(Vector input)
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

    protected virtual Vector ForwardForLearning(Vector input)
    {
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            RawLayerOutput[layer].MapIndexedInplace((index,x)=>input[index]);
            input.MapInplace(x=>layer.Activation.Activation(x));
        }
        return input;
    }
    /// <summary>
    /// Learns a model on error function. Use it when you don't have a dataset to train on.
    /// </summary>
    /// <param name="input">Given input to model</param>
    /// <param name="theta">Used to compute derivatives from error function. Bigger values allows to locate a local minima faster. In practice I use something like 0.01</param>
    /// <param name="errorFunction">
    /// Error function that must be implemented with following constraints: <br/>
    /// 1) It needs to be dependent on input vector.
    /// So when we put different input vector, it gives different results.<br/>
    /// 2) It needs to be continuous.
    /// So when model weights changes on some small theta, the output of error function 
    /// is also changing by some theta.<br/>
    /// 3) It need to use Forward method from given to it neural network parameter.
    /// </param>
    /// <returns></returns>
    public BackpropResult LearnOnError(Vector input,float theta, Func<Vector,NNBase,float> errorFunction){
        var original = errorFunction(input,this);
        var originalOutput = Forward(input);
        var errorDerivative = originalOutput.Map(x=>0.0f);
        Parallel.For(0,originalOutput.Count,i=>
        {
            var replacer = new ErrorFunctionOutputDerivativeReplacer
            {
                ChangedOutputIndex = i,
                ChangedOutputTheta = theta
            };
            var nn = new NNErrorDerivativeComputation(this,replacer);
            var changed = errorFunction(input,nn);
            errorDerivative[i]=(changed-original)/theta;
            replacer.ChangedOutputIndex = -1;
        });
        //fill layers with learning info
        ForwardForLearning(input);
        Learn(input,errorDerivative);
        return new BackpropResult(Layers);
    }

    public float Error(Vector input, Vector expected)
    {
        return (Forward(input) - expected).Sum(x => x * x);
    }
    
    void Learn(Vector input,Vector<float> errorDerivative)
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