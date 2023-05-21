using GradientDescentSharp.NeuralNetwork.WeightInitializers;
namespace GradientDescentSharp.NeuralNetwork;

public class ActivationFunction : IActivationFunction{
    private Func<float, float> activation;
    private Func<float, float> derivative;
    /// <summary>
    /// Epsilon that is used to compute activation function derivatives, in case
    /// when explicit derivative is not given
    /// </summary>
    /// <returns></returns>
    public static float Epsilon = 0.001f;
    public ActivationFunction(Func<float,float> activation, Func<float,float>? activationDerivative = null){
        this.activation = activation;
        this.derivative = activationDerivative ?? (x=>(activation(x+Epsilon)-activation(x))/Epsilon);
    }
    public static ActivationFunction Of(Func<float,float> activation, Func<float,float>? activationDerivative = null)
        => new(activation,activationDerivative);
    /// <summary>
    /// Use <see cref="Guassian"/> initializer
    /// </summary>
    public static ActivationFunction Linear(){
        return new(x=>x,x=>1);
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static ActivationFunction Sigmoid(){
        var sigmoid = (float x)=>(1.0f/(1+MathF.Exp(-x)));
        return new(sigmoid,x=>1-sigmoid(x));
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static ActivationFunction Softplus(){
        var sigmoid = (float x)=>(1.0f/(1+MathF.Exp(-x)));
        var softplus = (float x)=>MathF.Log(1+MathF.Exp(x));
        return new(softplus,sigmoid);
    }
    /// <summary>
    /// Use <see cref="HeNormal"/> or <see cref="He2Normal"/>  initializer
    /// </summary>
    public static ActivationFunction Relu(){
        return new(x=>Math.Max(0,x),x=>x>0 ? 1 : 0);
    }
    /// <summary>
    /// Use <see cref="He3Normal"/> initializer
    /// </summary>
    public static ActivationFunction Swish(float beta){
        var sigmoid = (float x)=>1.0f/(1+MathF.Exp(-x));
        var swish = (float x)=>x/(1+MathF.Exp(-beta*x));
        return new(swish,x=>{
            var s = swish(x);
            return s+sigmoid(beta*x)*1-s;
        });
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static ActivationFunction Tanh(){
        return new(x=>MathF.Tanh(x),x=>{
            var tanh = MathF.Tanh(x);
            return 1-tanh*tanh;
        });
    }
    public float Activation(float x) => activation(x);
    public float ActivationDerivative(float x) => derivative(x);
}