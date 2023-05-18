namespace GradientDescentSharp.NeuralNetwork;

record Learned(Vector biasesGradient, Vector layerInput, FloatType learningRate);

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
    public Vector ActivatedOutput => (Vector)RawOutput.Map(Activation.Activation);
    public IActivationFunction Activation{get;}
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
        RawOutput.MapIndexedInplace((index,_)=>raw[index]);
        return ActivatedOutput;
    }
    public void Learn(Vector biasesGradient, Vector layerInput, FloatType learningRate)
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
