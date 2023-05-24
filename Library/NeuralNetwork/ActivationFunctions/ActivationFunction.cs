using GradientDescentSharp.NeuralNetwork.WeightInitializers;
using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

public class ActivationFunction : IActivationFunction{
    private Func<Vector, Vector> activation;
    private Func<Vector, Vector> derivative;
    /// <summary>
    /// Epsilon that is used to compute activation function derivatives, in case
    /// when explicit derivative is not given
    /// </summary>
    /// <returns></returns>
    public static float Epsilon = 0.001f;
    public ActivationFunction(Func<Vector, Vector> activation, Func<Vector, Vector>? derivative  = null){
        this.activation = activation;
        var a = (Vector v)=>(Vector)((activation((Vector)(v + Epsilon)) -activation(v))/Epsilon);
        this.derivative = derivative ?? a;
    }
    public static ActivationFunction Of(Func<Vector,Vector> activation, Func<Vector,Vector>? activationDerivative = null)
        => new(activation,activationDerivative);
    /// <summary>
    /// Use <see cref="Guassian"/> initializer
    /// </summary>
    public static Linear Linear(){
        return new();
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static Sigmoid Sigmoid(){
        return new();
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static Softplus Softplus(){
        return new();
    }
    /// <summary>
    /// Use <see cref="HeNormal"/> or <see cref="He2Normal"/>  initializer
    /// </summary>
    public static Relu Relu(){
        return new();
    }
    /// <summary>
    /// Use <see cref="He3Normal"/> initializer
    /// </summary>
    public static Swish Swish(float beta){
        return new(beta);
      
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static Tanh Tanh(){
        return new();
    }
    /// <summary>
    /// Use <see cref="GlorotNormal"/> or <see cref="HeNormal"/> initializer
    /// </summary>
    public static Softmax Softmax(){
        return new();
    }
    public Vector Activation(Vector x) => activation(x);
    public Vector ActivationDerivative(Vector x) => derivative(x);
}
