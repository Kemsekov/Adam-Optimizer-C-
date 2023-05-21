
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

record Learned(Vector biasesGradient, Vector layerInput, float learningRate);

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
    public Layer(IComplexObjectsFactory<float> factory,int inputSize, int outputSize, IActivationFunction activation, IWeightsInit weightsInit){
        Weights = (Matrix)factory.CreateMatrix(outputSize,inputSize);
        Bias = (Vector)factory.CreateVector(outputSize);
        RawOutput = (Vector)factory.CreateVector(outputSize);
        Activation = activation;
        weightsInit.InitWeights(Bias);
        weightsInit.InitWeights(Weights);
    }
    public Vector Forward(Vector input){
        var raw = Weights*input+Bias;
        return (Vector)raw;
    }

    public void Learn(Vector biasesGradient, Vector layerInput, float learningRate)
    {
        
        // var weightsGradient = (int j,int k)=>biasesGradient[j]*layerInput[k];
        // Weights.MapIndexedInplace((j,k,x)=>x-learningRate*weightsGradient(j,k));

        for(int k = 0;k<layerInput.Count;k++){
            var kInput = layerInput[k];
            if(kInput==0) continue;
            for(int j = 0;j<Weights.RowCount;j++){
                Weights[j,k]-=learningRate*biasesGradient[j]*kInput;
            }
        }

        Bias.MapIndexedInplace((j,x)=>x-learningRate*biasesGradient[j]);
        learned = new Learned(biasesGradient,layerInput,learningRate);
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
