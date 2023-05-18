using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace GradientDescentSharp.NeuralNetwork;
public class ForwardNN : NNBase
{
    public ForwardNN(params Layer[] layers) : base(layers)
    {
    }
}