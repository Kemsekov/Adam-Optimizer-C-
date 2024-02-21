namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

/// <summary>
/// Contains weight initializers to get them into code faster
/// </summary>
public static class Initializers
{
    ///<inheritdoc cref="HeNormal"/>
    public static HeNormal HeNormal=>new();
    ///<inheritdoc cref="He2Normal"/>
    public static He2Normal He2Normal=>new();
    ///<inheritdoc cref="He3Normal"/>
    public static He3Normal He3Normal=>new();
    ///<inheritdoc cref="Guassian"/>
    public static Guassian Guassian=>new();
    ///<inheritdoc cref="GlorotUniform"/>
    public static GlorotUniform GlorotUniform=>new();
    ///<inheritdoc cref="NoInitialization"/>
    public static NoInitialization NoInitialization=>new();
}