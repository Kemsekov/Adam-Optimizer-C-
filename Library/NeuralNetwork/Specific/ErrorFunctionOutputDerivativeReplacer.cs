namespace GradientDescentSharp.NeuralNetwork.Specific;

/// <summary>
/// If you know how to do backpropagation, you know that you need to take a derivative 
/// in respect to forward method output on given input vector.
/// 
/// Now imagine that you don't have a derivative function for such task.
/// What to do?
/// 
/// The only way is to COMPUTE the derivative numerically, and 
/// you can do this only by passing around Forward method on input,
/// meanwhile retaining some theta change to output value model gives.
/// 
/// This is the reason this class exists.
/// 
/// I am sure there is more beautiful solution than this, but this is 
/// the best I could think of.
/// </summary>
public class LossFunctionOutputDerivativeReplacer{
    /// <summary>
    /// Index of output variable that will be replaced in derivative computation
    /// </summary>
    public int ChangedOutputIndex = -1;
    /// <summary>
    /// Epsilon in derivative computation
    /// </summary>
    public float ChangedOutputTheta;
    /// <summary>
    /// replaces some of output parameter by adding theta to it, so we could compute 
    /// derivative of output of neural network
    /// </summary>
    /// <param name="output"></param>
    public void ReplaceOutputParameter(FTensor output){
        if(ChangedOutputIndex!=-1)
            output[ChangedOutputIndex] += ChangedOutputTheta;
    }
}
