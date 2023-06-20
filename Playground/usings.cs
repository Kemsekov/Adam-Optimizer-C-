global using RentedArraySharp;
global using System.Collections;
global using GradientDescentSharp;
global using GradientDescentSharp.Utils;
global using GradientDescentSharp.DataAccessors;
global using GradientDescentSharp.GradientDescents;
global using GradientDescentSharp.ComplexDataStructures;
global using GradientDescentSharp.NeuralNetwork.WeightInitializers;
global using GradientDescentSharp.NeuralNetwork.ActivationFunction;
global using Playground;

public class A{
    public string Str = "";
    public static A operator+(A a1, A a2){
        return new A(){Str="AddedA"};
    }
}

public class B : A{
    public static B operator+(B a1, B a2){
        return new B(){Str="AddedB"};
    }
        public static B operator+(A a1, B a2){
        return new B(){Str="AddedB"};
    }
}