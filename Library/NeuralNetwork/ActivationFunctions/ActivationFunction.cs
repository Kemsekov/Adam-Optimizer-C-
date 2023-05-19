using GradientDescentSharp.NeuralNetwork.WeightInitializers;
namespace GradientDescentSharp.NeuralNetwork;

public class ActivationFunction : IActivationFunction{
    private Func<double, double> activation;
    private Func<double, double> derivative;
    /// <summary>
    /// Epsilon that is used to compute activation function derivatives, in case
    /// when explicit derivative is not given
    /// </summary>
    /// <returns></returns>
    public static double Epsilon = 0.0001;
    public ActivationFunction(Func<double,double> activation, Func<double,double>? activationDerivative = null){
        this.activation = activation;
        this.derivative = activationDerivative ?? (x=>(activation(x+Epsilon)-activation(x))/Epsilon);
    }
    public static ActivationFunction Of(Func<double,double> activation, Func<double,double>? activationDerivative = null)
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
        var sigmoid = (double x)=>(1.0/(1+Math.Exp(-x)));
        return new(sigmoid,x=>1-sigmoid(x));
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static ActivationFunction Softplus(){
        var sigmoid = (double x)=>(1.0/(1+Math.Exp(-x)));
        var softplus = (double x)=>Math.Log(1+Math.Exp(x));
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
    public static ActivationFunction Swish(double beta){
        var sigmoid = (double x)=>(1.0/(1+Math.Exp(-x)));
        var swish = (double x)=>(x/(1+Math.Exp(-beta*x)));
        return new(swish,x=>{
            var s = swish(x);
            return s+sigmoid(beta*x)*1-s;
        });
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static ActivationFunction Tanh(){
        return new(x=>Math.Tanh(x),x=>{
            var tanh = Math.Tanh(x);
            return 1-tanh*tanh;
        });
    }
    public double Activation(double x) => activation(x);
    public double ActivationDerivative(double x) => derivative(x);
}