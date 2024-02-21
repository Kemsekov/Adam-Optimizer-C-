/// <summary>
/// Regularization
/// </summary>
public interface IRegularization{
    /// <summary>
    /// Weight derivative for regularization.<br/>
    /// For l2 it is gonna be just 2*weight, for l1 it is gonna be |weight|/|weight|
    /// </summary>
    /// <param name="weight"></param>
    public float WeightDerivative(float weight);
}
/// <summary>
/// L2 regularization
/// </summary>
public record L2Regularization : IRegularization
{
    private float _alpha;
    /// <summary>
    /// </summary>
    /// <param name="alpha">Regularization coefficient</param>
    public L2Regularization(float alpha){
        this._alpha = alpha;
    }
    ///<inheritdoc/>
    public float WeightDerivative(float weight)
    {
        return _alpha*2*weight;
    }
}
/// <summary>
/// L1 regularization
/// </summary>
public record L1Regularization : IRegularization
{
    private float _alpha;
    /// <summary>
    /// </summary>
    /// <param name="alpha">Regularization coefficient</param>
    public L1Regularization(float alpha){
        this._alpha = alpha;
    }
    ///<inheritdoc/>
    public float WeightDerivative(float weight)
    {
        return _alpha*float.Sign(weight);
    }
}