using MathNet.Numerics.LinearAlgebra.Single;

namespace GradientDescentSharp.NeuralNetwork;

public interface ILearner{
    void Learn();
    void Unlearn();
}

/// <summary>
/// Learns something, and can unlearn it too
/// </summary>
public abstract record LearnerBase(LearningData LearningData) : ILearner
{
    protected ILayer layer => LearningData.layer;
    protected Vector biasesGradient => LearningData.biasesGradient;
    protected Vector layerInput => LearningData.layerInput;
    protected float learningRate => LearningData.learningRate;
    public abstract void Learn();
    public abstract void Unlearn();
}

public record LearningData(ILayer layer, Vector biasesGradient, Vector layerInput, float learningRate);