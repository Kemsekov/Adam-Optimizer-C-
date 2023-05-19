
namespace GradientDescentSharp.NeuralNetwork;

record Learned(Vector biasesGradient, Vector layerInput, double learningRate);

public class Layer : ILayer
{
    private Learned? learned;

    public Matrix Weights{get;}
    public Vector Bias{get;}
    /// <summary>
    /// Output without applied activation function. <br/>
    /// To get true output call
    /// <see cref="ActivatedOutput"/>
    /// </summary>
    /// <value></value>
    public Vector RawOutput{get;}
    public IActivationFunction Activation{get;}

    /// <param name="factory">Linear objects factory</param>
    /// <param name="inputSize">Layer input size</param>
    /// <param name="outputSize">Layer output size</param>
    /// <param name="activation">Activation function. May choose from <see cref="ActivationFunction"/></param>
    /// <param name="weightsInit">Weight initialization.</param>
    public Layer(IComplexObjectsFactory factory,int inputSize, int outputSize, IActivationFunction activation, IWeightsInit weightsInit){
        Weights = factory.CreateMatrix(outputSize,inputSize);
        Bias = factory.CreateVector(outputSize);
        RawOutput = factory.CreateVector(outputSize);
        Activation = activation;
        weightsInit.InitWeights(Bias);
        weightsInit.InitWeights(Weights);
    }
    public Vector Forward(Vector input){
        var raw = Weights*input+Bias;
        return (Vector)raw;
    }

    public void Learn(Vector biasesGradient, Vector layerInput, double learningRate)
    {
        var weightsGradient = (int j,int k)=>biasesGradient[j]*layerInput[k];
        Weights.MapIndexedInplace((j,k,x)=>x-learningRate*weightsGradient(j,k));
        Bias.MapIndexedInplace((j,x)=>x-learningRate*biasesGradient[j]);
        this.learned = new Learned(biasesGradient,layerInput,learningRate);
    }
    /// <summary>
    /// Unlearns last learned weights
    /// </summary>
    public void Unlearn(){
        if(learned is null) return;
        var biasesGradient = learned.biasesGradient;
        var layerInput = learned.layerInput;
        var learningRate = learned.learningRate;
        var weightsGradient = (int j,int k)=>biasesGradient[j]*layerInput[k];
        Weights.MapIndexedInplace((j,k,x)=>x+learningRate*weightsGradient(j,k));
        Bias.MapIndexedInplace((j,x)=>x+learningRate*biasesGradient[j]);
        learned = null;
    }
}
