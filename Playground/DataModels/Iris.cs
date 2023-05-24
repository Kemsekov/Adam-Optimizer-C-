using MathNet.Numerics.LinearAlgebra.Single;

namespace Playground.DataModels;
public record Iris
{
    public float sepal_length{get;set;}
    public float sepal_width{get;set;}
    public float petal_length{get;set;}
    public float petal_width{get;set;}
    public string species{get;set;}
    public static IDictionary<string,int> SpeciesMap = new Dictionary<string,int>(){
        {"Iris-setosa",0},
        {"Iris-versicolor",1},
        {"Iris-virginica",2}
    };
   public static IDictionary<int,string> IdToSpeciesMap = new Dictionary<int,string>(){
        {0,"Iris-setosa"},
        {1,"Iris-versicolor"},
        {2,"Iris-virginica"}
    };
    public Iris(){}

    public (Vector input, Vector output) BuildData(){
        var input = DenseVector.Create(4,0);
        input[0] = sepal_length;
        input[1] = sepal_width;
        input[2] = petal_length;
        input[3] = petal_width;

        var output = DenseVector.Create(3,0);
        output[SpeciesMap[species]]=1;
        return (input,output);
    }
}