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
    public static ActivationFunction Sigmoid(){
        var sigmoid = (FloatType x)=>1.0/(1+Math.Exp(-x));
        return new(sigmoid,x=>1-sigmoid(x));
    }
    public static ActivationFunction Softplus(){
        var sigmoid = (FloatType x)=>1.0/(1+Math.Exp(-x));
        var softplus = (FloatType x)=>Math.Log(1+Math.Exp(x));
        return new(softplus,sigmoid);
    }
    public static ActivationFunction Relu(){
        return new(x=>Math.Max(0,x),x=>x>0 ? 1 : 0);
    }
    public static ActivationFunction Swish(double beta){
        var sigmoid = (FloatType x)=>1.0/(1+Math.Exp(-x));
        return new(x=>x/(1+Math.Exp(-beta*x)));
    }
    public static ActivationFunction Tanh(){
        return new(Math.Tanh,x=>{
            var tanh = Math.Tanh(x);
            return 1-tanh*tanh;
        });
    }
    public double Activation(double x) => activation(x);
    public double ActivationDerivative(double x) => derivative(x);
}