using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork.WeightInitializers;

/// <summary>
/// Contains weight initializers to get them into code faster
/// </summary>
public static class Initializers
{
    public static HeNormal HeNormal=>new();
    public static He2Normal He2Normal=>new();
    public static He3Normal He3Normal=>new();
    public static Guassian Guassian=>new();
    public static GlorotUniform GlorotUniform=>new();
    public static GlorotNormal GlorotNormal=>new();
}