using System.Numerics;
using MathNet.Numerics.LinearAlgebra.Solvers;

namespace GradientDescentSharp.GradientDescents;

/// <summary>
/// Modified implementation of adam descent.<br/>
/// The sole difference is that whenever the next error value is bigger than previous,
/// the model rollback it's values, reduce the learning rate and step again.<br/>
/// It is guaranteed to find local minima.
/// </summary>
public abstract class AdamDescent<TFloat> : ReversibleLoggedGradientDescentBase<TFloat>
where TFloat : unmanaged,INumber<TFloat>
{
    RentedArray<TFloat> firstMomentum;
    RentedArray<TFloat> secondMomentum;
    /// <summary>
    /// Adam descent beta1 coefficient
    /// </summary>
    public TFloat Beta1;
    /// <summary>
    /// Adam descent beta2 coefficient
    /// </summary>
    public TFloat Beta2;
    /// <summary>
    /// Creates new instance of adam descent
    /// </summary>
    ///<inheritdoc/>
    public override TFloat DescentRateDecreaseRate{get;set;}
    /// <summary>
    /// Create new instance of adam descent
    /// </summary>
    public AdamDescent(IDataAccess<TFloat> variables, Func<IDataAccess<TFloat>, TFloat> function) : base(variables, function)
    {
        Beta1 = (TFloat)(0.9 as dynamic);
        Beta2 = (TFloat)(0.99 as dynamic);
        firstMomentum = ArrayPoolStorage.RentArray<TFloat>(Dimensions);
        secondMomentum = ArrayPoolStorage.RentArray<TFloat>(Dimensions);
        DescentRateDecreaseRate = TFloat.One - Beta1;
    }
    /// <summary>
    /// Computes sqrt from value
    /// </summary>
    protected abstract TFloat Sqrt(TFloat value);
    ///<inheritdoc/>
    protected override void ComputeChange(IDataAccess<TFloat> change, TFloat learningRate, TFloat currentEvaluation)
    {
        ComputeGradient(change,Variables, currentEvaluation);
        var gradient = change;
        int t = Iteration; // The timestep counter

        // Loop over the parameters
        Parallel.For(0,Dimensions,i=>
        {
            // Update the biased first moment estimate
            firstMomentum[i] = Beta1 * firstMomentum[i] + (TFloat.One - Beta1) * gradient[i];

            // Update the biased second moment estimate
            secondMomentum[i] = Beta2 * secondMomentum[i] + (TFloat.One - Beta2) * gradient[i] * gradient[i];
            // Compute the bias-corrected first moment estimate
            TFloat m_hat = firstMomentum[i] / (TFloat.One - Math<TFloat>.Pow(Beta1, t));

            // Compute the bias-corrected second moment estimate
            TFloat v_hat = secondMomentum[i] / (TFloat.One - Math<TFloat>.Pow(Beta2, t));

            // Update the parameters    
            change[i] = learningRate * m_hat / (Sqrt(v_hat) + Epsilon);
        });
    }

}
///<inheritdoc/>
public class AdamDescent : AdamDescent<double>
{
    ///<inheritdoc/>
    public AdamDescent(IDataAccess<double> variables, Func<IDataAccess<double>, double> function) : base(variables, function)
    {
    }
    ///<inheritdoc/>
    protected override double Sqrt(double value)=>Math.Sqrt(value);
}

///<inheritdoc/>
public class AdamDescentSingle : AdamDescent<float>
{
    ///<inheritdoc/>
    public AdamDescentSingle(IDataAccess<float> variables, Func<IDataAccess<float>, float> function) : base(variables, function)
    {
    }
    ///<inheritdoc/>
    protected override float Sqrt(float value)=>MathF.Sqrt(value);
}


