/// <summary>
/// Fisher information approximate implementation
/// </summary>
public class FisherInformationSingle
{
    /// <summary>
    /// Epsilon that is used to compute gradients
    /// </summary>
    public float Epsilon{get;set;} = (float)1e-7;
    /// <summary>
    /// Function that is used to generate some output from input samples
    /// </summary>
    public Func<float[], float> ModelFunction { get; }
    /// <summary>
    /// Function that is used to compute model gradients
    /// </summary>
    public Func<float[], float[]> Gradient { get; }

    /// <param name="fun">
    /// Function that takes n dim input and outputs some number. <br/>
    /// This may be your model.
    /// </param>
    /// <param name="gradient">
    /// Function that is used to compute model gradient from input data. 
    /// By default uses central difference approximation.<br/>
    /// Change <see cref="Epsilon"/> to adjust central difference gradient computation
    /// </param>
    public FisherInformationSingle(Func<float[], float> fun, Func<float[],float[]>? gradient = null)
    {
        ModelFunction = fun;
        Gradient = gradient ?? (x=>GradientCentralDifference(ModelFunction,x,Epsilon));
    }
    /// <summary>
    /// Computes fisher information over one sample of data.<br/>
    /// It computes how much changing <paramref name="sample"/> 
    /// variables affects model's output. <br/>
    /// So for example if your sample is cat's image and <see cref="ModelFunction"/>
    /// is model that determines whether given image is cat or dog, 
    /// you may expect high fisher information values for cats and dogs images
    /// and small fisher information values for some random white-noise images.<br/>
    /// It is useful to determine "important" samples of your data set that have
    /// a lot of value for model learning.
    /// </summary>
    /// <param name="sample">Data sample</param>
    /// <returns>
    /// Fisher information for each of sample's data parameters.<br/>
    /// Null if parameters contains NaN.<br/>
    /// Normalize it with parameter 1 on this method output to get percentage
    /// of parameter influence on model function<br/>
    /// Get length of it to determine how "valuable" or "important" that sample
    /// to your model.
    /// </returns>
    public virtual float[]? OverSample(float[] sample)
    {
        var sampleClone = sample.ToArray();
        var dims = sample.Length;
        var gradient = Gradient(sample);
        if (gradient.Any(float.IsNaN)) return null;
        for(int i = 0;i<dims;i++)
            gradient[i]*=gradient[i];
        //squared gradient
        return gradient;
    }
    /// <param name="samples">
    /// Samples that will be used to compute fisher information
    /// </param>
    ///<inheritdoc cref="OverGenerated"/>
    public float[] OverSamples(IEnumerable<float[]> samples)
    {
        var result = samples.First().ToArray();
        Array.Fill(result, 0);
        var dims = result.Length;
        var samplesArray = samples.ToArray();
        Parallel.ForEach(samplesArray, input =>
        {
            var squaredGradient = OverSample(input);
            if (squaredGradient is null) return;
            lock (result)
            {
                for (int k = 0; k < result.Length; k++)
                    result[k] += squaredGradient[k];
            }
        });
        for (int k = 0; k < result.Length; k++)
            result[k] /= samplesArray.Length;
        return result;
    }
    /// <summary>
    /// Approximates fisher information of function over some randomly generated inputs.
    /// </summary>
    /// <param name="generateSample">
    /// Your function is defined on some interval, so generate some possible 
    /// random uniform function inputs on which this function is determined.<br/>
    /// Alternatively, you can returns some samples from your dataset.<br/>
    /// This values will be used to approximate fisher information.
    /// </param>
    /// <param name="expectationsSampleCount">
    /// How many samples to generate. 
    /// The more - the more precise approximation
    /// </param>
    /// <returns>
    /// Fisher information for each of data parameters.
    ///  Normalize it with parameter 1 on this method output to get percentage
    ///  of parameter influence on problem function
    /// </returns>
    public float[] OverGenerated(Func<float[]> generateSample, int expectationsSampleCount)
    {
        var result = generateSample().ToArray();
        Array.Fill(result, 0);

        var dims = result.Length;
        Parallel.For(0, expectationsSampleCount, i =>
        {
            var input = generateSample();
            var squaredGradient = OverSample(input);
            if (squaredGradient is null) return;
            lock (result)
            {
                for (int k = 0; k < result.Length; k++)
                    result[k] += squaredGradient[k];
            }
        });
        for (int k = 0; k < result.Length; k++)
            result[k] /= expectationsSampleCount;
        return result;
    }
    /// <summary>
    /// Central difference approximation.<br/>
    /// Do not use it on same input vector in multiple threads! <br/>
    /// Create copy of input data for each thread instead. Use <see cref="ThreadLocal{T}"/>
    /// </summary>
    public static float CentralDifference(Func<float[], float> func, float[] input, int paramIndex, float epsilon)
    {
        input[paramIndex] += epsilon;
        var left = func(input);
        input[paramIndex] -= 2 * epsilon;
        var right = func(input);
        return (left - right) * 0.5f / epsilon;
    }
    /// <summary>
    /// Approximates gradient of a function using central difference
    /// </summary>
    public static float[] GradientCentralDifference(Func<float[], float> func, float[] sample,float epsilon)
    {
        var sampleClone = sample.ToArray();
        var dims = sample.Length;
        var gradient =
            Enumerable.Range(0, dims)
            .Select(i => CentralDifference(func, sampleClone, i, epsilon))
            .ToArray();
        return gradient;
    }
}
