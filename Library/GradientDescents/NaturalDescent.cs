using System.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace GradientDescentSharp.GradientDescents;

/// <summary>
/// Performs natural descent using fisher information matrix
/// </summary>
public abstract class NaturalDescent<TFloat> : ReversibleLoggedGradientDescentBase<TFloat>
where TFloat : unmanaged, INumber<TFloat>
{
    Func<IDataAccess<TFloat>, TFloat> likelihood;
    /// <summary>
    /// Used to randomly sample given parameters.<br/>
    /// When computing fisher information, we will need to compute expectation over
    /// some range of parameter space over some function, so this method is used
    /// to generate parameters.
    /// </summary>
    public Func<int, TFloat> GenerateParameterSample;
    /// <summary>
    /// How much decrease descent rate when we step into bigger error value.<br/>
    /// By default it is 0.1, so when we step into worse error function value,
    /// we will divide  learning rate by 10.
    /// </summary>
    public override TFloat DescentRateDecreaseRate{get;set;} = (TFloat)(0.1 as dynamic);
    /// <summary>
    /// To compute expectation for fisher information, we need to generate a range of samples.
    /// This parameter describes how many of them to generate.<br/> 
    /// General rule is that count of samples should grow
    /// exponentially proportional to parameters dimensions
    /// </summary>
    public int ExpectationsSampleCount = 100;
    Lazy<Matrix<TFloat>> fisherInformationMatrix;
    /// <summary>
    /// Inverse of fisher information matrix
    /// </summary>
    public Matrix<TFloat> FisherInformationMatrixInverse => fisherInformationMatrix.Value;
    /// <summary>
    /// Method to get log of value
    /// </summary>
    protected abstract TFloat Log(TFloat value);
    /// <summary>
    /// Returns random float
    /// </summary>
    protected abstract TFloat Random();
    /// <summary>
    /// Returns empty vector of length
    /// </summary>
    protected abstract MathNet.Numerics.LinearAlgebra.Vector<TFloat> EmptyVector(int length);
    /// <summary>
    /// Returns empty matrix of given size
    /// </summary>
    protected abstract MathNet.Numerics.LinearAlgebra.Matrix<TFloat> EmptyMatrix(int rows,int cols);
    /// <summary>
    /// Creates complex objects factory of given TFloat type
    /// </summary>
    protected abstract IComplexObjectsFactory<TFloat> ComplexObjectsFactory(IDataAccess<TFloat> data);
    /// <summary>
    /// Creates natural descent <see cref="NaturalDescent"/> 
    /// </summary>
    ///<inheritdoc/>    
    public NaturalDescent(IDataAccess<TFloat> variables, Func<IDataAccess<TFloat>, TFloat> function) : base(variables, function)
    {
        //to transform error function to likelihood function, use following mapping
        likelihood = (IDataAccess<TFloat> x) =>
        {
            var err = function(x);
            // return Math.Exp(-err) * (err + 1) / 2;
            return Log(err+TFloat.One)-err;
        };
        GenerateParameterSample = i => Random();
        ExpectationsSampleCount = (int)Math.Pow(10,variables.Length);
        fisherInformationMatrix = new(()=>ComputeFisherInformationMatrix(ExpectationsSampleCount).Inverse());
    }
    /// <summary>
    /// Computes fisher information matrix using monte-carlo approximation.<br/>
    /// It generates <see cref="ExpectationsSampleCount"/> samples and returns
    /// computed information matrix as average of sample results.
    /// </summary>
    /// <returns></returns>
    public Matrix<TFloat> ComputeFisherInformationMatrix(int expectationsSampleCount)
    {
        expectationsSampleCount = expectationsSampleCount==0 ? ExpectationsSampleCount : expectationsSampleCount;
        var matrix = EmptyMatrix(Dimensions,Dimensions);
        Parallel.For(0, expectationsSampleCount, k =>
        {
            using var arr = ArrayPoolStorage.RentArray<TFloat>(Dimensions);
            var variables = new RentedArrayDataAccess<TFloat>(arr);
            for (int i = 0; i < variables.Length; i++)
            {
                variables[i] = GenerateParameterSample(i);
            }

            using var derivative = derivativeOfLikelihood(variables);
            if(derivative.Any(TFloat.IsNaN)) return;
            lock (matrix)
                for (int i = 0; i < Dimensions; i++)
                {
                    var iDerivative = derivative[i];
                    for (int j = 0; j < Dimensions; j++)
                    {
                        var jDerivative = derivative[j];
                        matrix[i, j] += iDerivative * jDerivative;
                    }
                }
        });
        var convertedExpectationsSampleCount = (TFloat)(expectationsSampleCount as dynamic);

        matrix.MapInplace(x => x / convertedExpectationsSampleCount);
        return matrix;
    }
    /// <summary>
    /// This method computes derivative of likelihood function,
    /// which is used for computing fisher information matrix
    /// </summary>
    RentedArray<TFloat> derivativeOfLikelihood(IDataAccess<TFloat> Variables)
    {
        var derivativeOfLikelihood = ArrayPoolStorage.RentArray<TFloat>(Variables.Length);
        var c = likelihood(Variables);
        for (int i = 0; i < Variables.Length; i++)
        {
            Variables[i] += Epsilon;
            var b = likelihood(Variables);
            Variables[i] -= Epsilon;
            derivativeOfLikelihood[i] = (b - c) / Epsilon;
        };
        return derivativeOfLikelihood;
    }
    ///<inheritdoc/>
    protected override void ComputeChange(IDataAccess<TFloat> change, TFloat learningRate, TFloat currentEvaluation)
    {
        ComputeGradient(change,Variables, currentEvaluation);
        var gradientVector = ComplexObjectsFactory(change).CreateVector(change.Length);
        var result = learningRate*FisherInformationMatrixInverse*gradientVector;
        for(int i = 0;i<change.Length;i++)
            change[i] = result[i];
    }
}

///<inheritdoc/>
public class NaturalDescent : NaturalDescent<double>
{
    ///<inheritdoc/>
    public NaturalDescent(IDataAccess<double> variables, Func<IDataAccess<double>, double> function) : base(variables, function)
    {
    }

    ///<inheritdoc/>
    protected override IComplexObjectsFactory<double> ComplexObjectsFactory(IDataAccess<double> data)
    {
        return new ComplexObjectsFactory(data);
    }

    ///<inheritdoc/>
    protected override DMatrix EmptyMatrix(int rows, int cols)
    {
        return MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Create(rows,cols,0);
    }

    ///<inheritdoc/>
    protected override DVector EmptyVector(int length)
    {
        return MathNet.Numerics.LinearAlgebra.Double.DenseVector.Create(length,0);
    }

    ///<inheritdoc/>
    protected override double Log(double value)
    {
        return Math.Log(value);
    }

    ///<inheritdoc/>
    protected override double Random()
    {
        return System.Random.Shared.NextDouble()*2-1;
    }
}

    ///<inheritdoc/>
public class NaturalDescentSingle : NaturalDescent<float>
{
    ///<inheritdoc/>
    public NaturalDescentSingle(IDataAccess<float> variables, Func<IDataAccess<float>, float> function) : base(variables, function)
    {
    }

    ///<inheritdoc/>
    protected override IComplexObjectsFactory<float> ComplexObjectsFactory(IDataAccess<float> data)
    {
        return new ComplexObjectsFactorySingle(data);
    }

    ///<inheritdoc/>
    protected override FMatrix EmptyMatrix(int rows, int cols)
    {
        return MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.Create(rows,cols,0);
    }

    ///<inheritdoc/>
    protected override FVector EmptyVector(int length)
    {
        return MathNet.Numerics.LinearAlgebra.Single.DenseVector.Create(length,0);
    }

    ///<inheritdoc/>
    protected override float Log(float value)
    {
        return MathF.Log(value);
    }

    ///<inheritdoc/>
    protected override float Random()
    {
        return System.Random.Shared.NextSingle()*2-1;
    }
}
