using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Swish : IActivationFunction
{
    private float beta;

    public Swish(float beta){
        this.beta = beta;
    }
    public FVector Activation(FVector x)
    {
        return (FVector)x.Map(x=>x/(1+MathF.Exp(-beta*x)));
    }

    public FVector ActivationDerivative(FVector x)
    {
        return (FVector)x.Map(x=>{
            x*=beta;
            var sigmoid = 1.0f/(1+MathF.Exp(-x));
            var sigmoidDerivative = 1-sigmoid;
            return sigmoid+x*sigmoidDerivative;
        });
    }
}
