using GradientDescentSharp.NeuralNetwork.Specific;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

/// <summary>
/// Basic neural network implementation, with some additional features
/// </summary>
public abstract class NNBase
{
    /// <summary>
    /// Network layers
    /// </summary>
    public ILayer[] Layers { get; }
    /// <summary>
    /// Network learning rate
    /// </summary>
    public float LearningRate = 0.05f;
    /// <summary>
    /// Raw layer output from learning forward method, used for training
    /// </summary>
    protected Dictionary<ILayer, Vector> RawLayerOutput;

    /// <summary>
    /// Builds a copy of another neural network
    /// </summary>
    /// <param name="another"></param>
    public NNBase(NNBase another)
    {
        Layers = another.Layers;
        RawLayerOutput = another.RawLayerOutput;
    }
    /// <summary>
    /// Creates a new neural network from given layers
    /// </summary>
    public NNBase(params ILayer[] layers)
    {
        Layers = layers;
        RawLayerOutput = new Dictionary<ILayer, Vector>();
        foreach (var layer in layers)
        {
            RawLayerOutput[layer] = (Vector)layer.Bias.Map(x => 0.0f);
        }
    }
    /// <returns>Model prediction</returns>
    public virtual Vector Forward(Vector input)
    {
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            input.MapInplace(x => layer.Activation.Activation(x));
        }
        return input;
    }
    /// <summary>
    /// Does same forward, but keeps intermediate values to use it for training later
    /// </summary>
    protected virtual Vector ForwardForLearning(Vector input)
    {
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            RawLayerOutput[layer].MapIndexedInplace((index, x) => input[index]);
            input.MapInplace(x => layer.Activation.Activation(x));
        }
        return input;
    }
    /// <summary>
    /// Replaces saturated weights with new one from same weights distribution, but tries
    /// to keep weight mean on it's original value.<br/>
    /// Does not affect biases.<br/>
    /// Use this method when your model meets new kind of data, and needs to be retrained, but
    /// cannot because model weights is too saturated.
    /// </summary>
    /// <param name="weightsCount">How many weights to regenerate, left it -1 to regenerate all weights that smaller than min value or bigger than max value</param>
    /// <param name="minValue">Weights smaller than this value will be regenerated</param>
    /// <param name="maxValue">Weights bigger than this value will be regenerated</param>
    /// <param name="variation">How big change from previous weight value we need to make</param>
    /// <returns>Count of weights replaced</returns>
    public int RegenerateSaturatedWeights(int weightsCount = -1, float minValue = 0.01f, float maxValue = 0.99f, float variation = 1)
    {
        int count = 0;
        foreach (var layer in Layers)
        {
            var weights = layer.Weights;
            for (int i = 0; i < layer.Weights.RowCount; i++)
                for (int j = 0; j < layer.Weights.ColumnCount; j++)
                {
                    var weight = weights[i, j];
                    var sample = layer.WeightsInit.SampleWeight(weights);
                    if (weight > maxValue || weight < minValue)
                    {
                        weights[i, j] = weight + sample * variation;
                        count++;
                    }
                    if (weightsCount > 0 && count > weightsCount) return count;
                }
        }
        return count;
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
    /// 3) It need to use Forward method(maybe even several times) from given to it neural network parameter.<br/>
    /// </param>
    /// <returns></returns>
    public BackpropResult LearnOnError(Vector input, float theta, Func<Vector, PredictOnlyNN, float> errorFunction)
    {
        var errorDerivative = ComputeDerivativeOfErrorFunction(input, theta, errorFunction);

        //fill layers with learning info
        ForwardForLearning(input);
        var learned = Learn(input, errorDerivative);
        return new BackpropResult(learned);
    }

    private Vector<float> ComputeDerivativeOfErrorFunction(Vector input, float theta, Func<Vector, PredictOnlyNN, float> errorFunction)
    {
        var original = errorFunction(input, new(this));
        var originalOutput = Forward(input);
        var errorDerivative = originalOutput.Map(x => 0.0f);

        void computeDerivative(int i)
        {
            var replacer = new ErrorFunctionOutputDerivativeReplacer
            {
                ChangedOutputIndex = i,
                ChangedOutputTheta = theta
            };
            var nn = new NNErrorDerivativeComputation(this, replacer);
            var changed = errorFunction(input, new(nn));
            errorDerivative[i] = (changed - original) / theta;
            replacer.ChangedOutputIndex = -1;
        }

        //If we predicting one single value, it does not
        //make sense to parallelize it

        if (originalOutput.Count > 1)
            Parallel.For(0, originalOutput.Count, i =>
            {
                computeDerivative(i);
            });
        else
            computeDerivative(0);
        return errorDerivative;
    }

    /// <returns>MSE error from input and expected value from model prediction</returns>
    public float Error(Vector input, Vector expected)
    {
        return (Forward(input) - expected).Sum(x => x * x);
    }
    IEnumerable<Learned> Learn(Vector input, Vector<float> errorDerivative)
    {
        var learned = new List<Learned>();
        for (int i = Layers.Length - 1; i >= 0; i--)
        {
            var layer = Layers[i];

            var biases = layer.Bias;
            var layerOutput = RawLayerOutput[layer];
            var activation = layer.Activation.Activation;
            var derivative = layer.Activation.ActivationDerivative;

            var biasesGradient = (Vector)layerOutput.MapIndexed((j, x) => derivative(x) * errorDerivative[j]);
            if (i > 0)
            {
                errorDerivative.MapIndexedInplace((j, x) => x * derivative(layerOutput[j]));
                errorDerivative *= layer.Weights;
            }
            var layerInput = (Vector)(i > 0 ? RawLayerOutput[Layers[i - 1]].Map(activation) : input);

            //here we update weights

            for (int k = 0; k < layerInput.Count; k++)
            {
                var kInput = layerInput[k];
                if (kInput == 0) continue;
                for (int j = 0; j < layer.Weights.RowCount; j++)
                {
                    var gradient = biasesGradient[j] * kInput;
                    layer.Weights[j, k] -= LearningRate * gradient;
                }
            }

            layer.Bias.MapIndexedInplace((j, x) => x - LearningRate * biasesGradient[j]);

            learned.Add(new Learned(layer,biasesGradient, layerInput, LearningRate));
        }
        return learned;
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
        var learned = Learn(input, error);
        return new BackpropResult(learned);
    }
    ///<inheritdoc/>
    public static implicit operator PredictOnlyNN(NNBase t) => new(t);
}