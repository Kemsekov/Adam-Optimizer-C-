namespace GradientDescentSharp.NeuralNetwork.ActivationFunction;

/// <summary>
/// Contains a list of predefined activation functions
/// </summary>
public class ActivationFunction{
    
    /// <summary>
    /// Use <see cref="Guassian"/> initializer
    /// </summary>
    public static Linear Linear(){
        return new();
    }
    /// <summary>
    /// Use <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static Sigmoid Sigmoid(){
        return new();
    }
    /// <summary>
    /// Use <see cref="GlorotUniform"/> initializer
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
    /// Use <see cref="HeNormal"/> or <see cref="He2Normal"/>  initializer
    /// </summary>
    public static LeakyRelu LeakyRelu(float alpha){
        return new(alpha);
    }
    /// <summary>
    /// Use <see cref="He3Normal"/> initializer
    /// </summary>
    public static Swish Swish(float beta){
        return new(beta);
      
    }
    /// <summary>
    /// Use <see cref="GlorotUniform"/> initializer
    /// </summary>
    public static Tanh Tanh(){
        return new();
    }
    /// <summary>
    /// Use <see cref="HeNormal"/> initializer
    /// </summary>
    public static Softmax Softmax(){
        return new();
    }
}
