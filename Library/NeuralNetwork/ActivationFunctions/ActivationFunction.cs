using GradientDescentSharp.NeuralNetwork.WeightInitializers;
namespace GradientDescentSharp.NeuralNetwork;

public class ActivationFunction : IActivationFunction{
    private Func<double, double> activation;
    private Func<double, double> derivative;

    public ActivationFunction(Func<FloatType,FloatType> activation, Func<FloatType,FloatType>? activationDerivative = null){
        var epsilon = 0.0001;
        this.activation = activation;
        this.derivative = activationDerivative ?? (x=>(activation(x+epsilon)-activation(x))/epsilon);
    }
    public static ActivationFunction Of(Func<FloatType,FloatType> activation, Func<FloatType,FloatType>? activationDerivative = null)
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
        var sigmoid = (FloatType x)=>1.0/(1+Math.Exp(-x));
        return new(sigmoid,x=>1-sigmoid(x));
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static ActivationFunction Softplus(){
        var sigmoid = (FloatType x)=>1.0/(1+Math.Exp(-x));
        var softplus = (FloatType x)=>Math.Log(1+Math.Exp(x));
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
    public static ActivationFunction Swish(FloatType beta){
        var sigmoid = (FloatType x)=>1.0/(1+Math.Exp(-x));
        var swish = (FloatType x)=>x/(1+Math.Exp(-beta*x));
        return new(swish,x=>{
            var s = swish(x);
            return s+sigmoid(beta*x)*1-s;
        });
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static ActivationFunction Tanh(){
        return new(Math.Tanh,x=>{
            var tanh = Math.Tanh(x);
            return 1-tanh*tanh;
        });
    }
    public double Activation(double x) => activation(x);
    public double ActivationDerivative(double x) => derivative(x);
}