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
    /// Raw layer outputs from learning forward method, used for training
    /// </summary>
    protected ObjectPool<Dictionary<ILayer, Vector>> RawLayerOutputStorage { get; }

    /// <summary>
    /// Network learning rate
    /// </summary>
    public float LearningRate = 0.05f;

    /// <summary>
    /// Builds a copy of another neural network
    /// </summary>
    /// <param name="another"></param>
    public NNBase(NNBase another)
    {
        Layers = another.Layers;
        RawLayerOutputStorage = another.RawLayerOutputStorage;
    }
    /// <summary>
    /// Creates a new neural network from given layers
    /// </summary>
    public NNBase(params ILayer[] layers)
    {
        Layers = layers;
        this.RawLayerOutputStorage = new ObjectPool<Dictionary<ILayer, Vector>>(()=>{
            var rawLayerOutput = new Dictionary<ILayer, Vector>();
            foreach (var layer in layers)
            {
                rawLayerOutput[layer] = (Vector)layer.Bias.Map(x => 0.0f);
            }
            return rawLayerOutput;
        });
    }
    /// <returns>Model prediction</returns>
    public virtual Vector Forward(Vector input)
    {
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Activation.Activation(layer.Forward(input));
        }
        return input;
    }
    /// <summary>
    /// Does same forward, but keeps intermediate values to use it for training later
    /// </summary>
    protected virtual Vector ForwardForLearning(Vector input, out Dictionary<ILayer, Vector> rawLayerOutput)
    {
        rawLayerOutput = RawLayerOutputStorage.GetObject();
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            rawLayerOutput[layer].MapIndexedInplace((index, x) => input[index]);
            input=layer.Activation.Activation(input);
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
    /// <returns>Count of replaced weights</returns>
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
    /// Learns a model on loss function. Use it when you don't have a dataset to train on.
    /// </summary>
    /// <param name="input">Given input to model</param>
    /// <param name="theta">Used to compute derivatives from error function. Bigger values allows to locate a local minima faster. In practice I use something like 0.01</param>
    /// <param name="lossFunction">
    /// Loss function that must be implemented with following constraints: <br/>
    /// 1) It needs to be dependent on input vector.
    /// So when we put different input vector, it gives different results.<br/>
    /// 2) It needs to be continuous.
    /// So when model weights changes on some small theta, the output of error function 
    /// is also changing by some theta.<br/>
    /// 3) It need to use Forward method(maybe even several times) from given to it neural network parameter.<br/>
    /// </param>
    /// <returns>BackpropResult that can be used to apply computed gradients</returns>
    public BackpropResult LearnOnLoss(Vector input, float theta, Func<Vector, PredictOnlyNN, float> lossFunction)
    {
        var errorDerivative = ComputeDerivativeOfLossFunction(input, theta, lossFunction, out var rawLayerOutput);
        var learned = BuildLearner(ComputeGradients(input,errorDerivative,rawLayerOutput));
        RawLayerOutputStorage.PutObject(rawLayerOutput);
        return new BackpropResult(learned);
    }

    /// <summary>
    /// Default backpropagation implementation. Learns a model to predict given expected value from input
    /// </summary>
    /// <returns>BackpropResult that can be used to apply computed gradients</returns>
    public BackpropResult Backwards(Vector input, Vector expected)
    {
        // compute error derivative for MSE
        var error = (ForwardForLearning(input,out var rawLayersOutput) - expected) * 2;
        var gradients = ComputeGradients(input,error, rawLayersOutput);
        var learner = BuildLearner(gradients);
        RawLayerOutputStorage.PutObject(rawLayersOutput);
        return new BackpropResult(learner);
    }
    /// <returns>MSE error from input and expected value from model prediction</returns>
    public float Error(Vector input, Vector expected)
    {
        return (Forward(input) - expected).Sum(x => x * x);
    }
    private Vector<float> ComputeDerivativeOfLossFunction(Vector input, float theta, Func<Vector, PredictOnlyNN, float> errorFunction, out Dictionary<ILayer, Vector> rawLayerOutput)
    {
        var original = errorFunction(input, new(this));
        var originalOutput = ForwardForLearning(input,out rawLayerOutput);
        var errorDerivative = originalOutput.Map(x => 0.0f);

        void computeDerivative(int i)
        {
            var replacer = new LossFunctionOutputDerivativeReplacer
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


    
    record Gradient(int layerId, Vector<float> biasesGradients, Vector<float> layerInput );
    Gradient[] ComputeGradients(Vector input, Vector<float> errorDerivative, Dictionary<ILayer, Vector> rawLayersOutput){
        var result = new Gradient[Layers.Length];
        for (int i = Layers.Length - 1; i >= 0; i--)
        {
            var layer = Layers[i];

            var biases = layer.Bias;
            var layerOutput = rawLayersOutput[layer];
            var activation = layer.Activation.Activation;
            var derivative = layer.Activation.ActivationDerivative;

            // var biasesGradient = layerOutput.MapIndexed((j, x) => derivative(x) * errorDerivative[j]);
            var layerOutputDerivative = derivative(layerOutput);
            var biasesGradient = layerOutputDerivative.MapIndexed((j,x)=>x*errorDerivative[j]);

            if (i > 0)
            {
                errorDerivative.MapIndexedInplace((j, x) => x * layerOutputDerivative[j]);
                errorDerivative *= layer.Weights;
            }
            var layerInput = i > 0 ? activation(rawLayersOutput[Layers[i - 1]]) : input;

            //here we update weights
            result[i] = new(i,biasesGradient,layerInput);
        }
        return result;
    }
    
    IEnumerable<Learner> BuildLearner(Gradient[] gradients)
    {
        var learned = new List<Learner>();
        
        foreach(var layerInfo in gradients)
        {
            var layer = Layers[layerInfo.layerId];
            var layerInput = layerInfo.layerInput;
            var biasesGradient = layerInfo.biasesGradients;

            learned.Add(new Learner(layer, (Vector)biasesGradient, (Vector)layerInput.Clone(), LearningRate));
        }
        return learned;
    }

    ///<inheritdoc/>
    public static implicit operator PredictOnlyNN(NNBase t) => new(t);
}