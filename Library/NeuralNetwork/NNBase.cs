using GradientDescentSharp.NeuralNetwork.Specific;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;
public record Gradient(FVector biasesGradients, FVector layerInput );
/// <summary>
/// Basic neural network implementation, with some additional features
/// </summary>
public abstract class NNBase
{
    /// <summary>
    /// Learner factory. You can change it to use different gradient applying algorithms.<br/>
    /// By default <see cref="DefaultLearner"/> is used, which just applies gradient multiplying it with learning rate
    /// </summary>
    /// <returns></returns>
    public Func<LearningData,ILearner> LearnerFactory{get;set;} = DefaultLearner.Factory();
    /// <summary>
    /// Network layers
    /// </summary>
    public ILayer[] Layers { get; }

    /// <summary>
    /// Raw layer outputs from learning forward method, used for training
    /// </summary>
    protected ObjectPool<Dictionary<ILayer, FVector>> RawLayerOutputStorage { get; }

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
        this.RawLayerOutputStorage = new ObjectPool<Dictionary<ILayer, FVector>>(()=>{
            var rawLayerOutput = new Dictionary<ILayer, FVector>();
            foreach (var layer in layers)
            {
                rawLayerOutput[layer] = layer.Bias.Map(x => 0.0f);
            }
            return rawLayerOutput;
        });
    }
    /// <returns>Model prediction</returns>
    public virtual FVector Forward(FVector input)
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
    protected virtual FVector ForwardForLearning(FVector input, out Dictionary<ILayer, FVector> rawLayerOutput)
    {
        rawLayerOutput = RawLayerOutputStorage.GetObject();
        ILayer layer;
        for (int i = 0; i < Layers.Length; i++)
        {
            layer = Layers[i];
            input = layer.Forward(input);
            rawLayerOutput[layer]=input;
            input=layer.Activation.Activation(input);
        }
        return input;
    }
    /// <summary>
    /// Replaces saturated weights with new one from same weights distribution, but tries
    /// to keep weight mean on it's original value.<br/>
    /// Does not affect biases.<br/>
    /// Use this method when your model meets new kind of data, and needs to be retrained, but
    /// cannot because model weights is too saturated.<br/>
    /// Causes catastrophic forgetting too, so be cautious.
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
    public BackpropResult LearnOnLoss(FVector input, float theta, Func<FVector, PredictOnlyNN, float> lossFunction)
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
    public BackpropResult Backwards(FVector input, FVector expected)
    {
        // compute error derivative for MSE
        var error = (ForwardForLearning(input,out var rawLayersOutput) - expected) * 2;
        var gradients = ComputeGradients(input,error, rawLayersOutput);
        var learner = BuildLearner(gradients);
        RawLayerOutputStorage.PutObject(rawLayersOutput);
        return new BackpropResult(learner);
    }
    /// <returns>MSE error from input and expected value from model prediction</returns>
    public float Error(FVector input, FVector expected)
    {
        return (Forward(input) - expected).Sum(x => x * x);
    }
    private FVector ComputeDerivativeOfLossFunction(FVector input, float theta, Func<FVector, PredictOnlyNN, float> errorFunction, out Dictionary<ILayer, FVector> rawLayerOutput)
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

    Gradient[] ComputeGradients(FVector input, FVector lossDerivative, Dictionary<ILayer, FVector> rawLayersOutput){
        var result = new Gradient[Layers.Length];
        for (int i = Layers.Length - 1; i >= 0; i--)
        {
            var layer = Layers[i];
            var layerOutput = rawLayersOutput[layer];
            bool updateLossDerivative = i>0;
            var layerInput = i > 0 ? layer.Activation.Activation(rawLayersOutput[Layers[i - 1]]) : input.Clone();
            result[i] = layer.ComputeGradient(layerInput,layerOutput,lossDerivative,updateLossDerivative,out var newLoss);
            if(newLoss is not null)
                lossDerivative = newLoss;
        }
        return result;
    }
    
    IEnumerable<DefaultLearner> BuildLearner(Gradient[] gradients)
    {
        var learned = new List<DefaultLearner>();
        
        foreach(var layerInfo in Layers.Zip(gradients))
        {
            var layer = layerInfo.First;
            var layerInput = layerInfo.Second.layerInput;
            var biasesGradient = layerInfo.Second.biasesGradients;
            var data = new LearningData(layer, biasesGradient, layerInput, LearningRate);
            learned.Add(new DefaultLearner(data));
        }
        return learned;
    }

    ///<inheritdoc/>
    public static implicit operator PredictOnlyNN(NNBase t) => new(t);
}