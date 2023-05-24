using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class Swish : IActivationFunction
{
    private float beta;

    public Swish(float beta){
        this.beta = beta;
    }
    public Vector Activation(Vector x)
    {
        return (Vector)x.Map(x=>x/(1+MathF.Exp(-beta*x)));
    }

    public Vector ActivationDerivative(Vector x)
    {
        return (Vector)x.Map(x=>{
            x*=beta;
            var sigmoid = 1.0f/(1+MathF.Exp(-x));
            var sigmoidDerivative = 1-sigmoid;
            return sigmoid+x*sigmoidDerivative;
        });
    }
}
