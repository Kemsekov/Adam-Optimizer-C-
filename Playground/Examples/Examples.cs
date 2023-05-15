namespace Playground;
public partial class Examples
{
    /// <summary>
    /// Just a set of functions that can be minimized
    /// </summary>
    public static Func<IDataAccess<double>, double>[] Functions ={
                (d)=>Math.Abs(d[0]-d[1]*d[2]+Math.Exp(d[0]*d[1]*d[2])-Math.Sin(d[0]+d[1]+d[2])),
                (d)=>Math.Abs(d[0]-d[1]*d[2]),
                (d)=>Math.Abs(d[0]*d[1]*d[2]-d[0]*Math.Sin(d[2])+Math.Cos(d[0]*d[2])+d[2]),
                (d)=>Math.Abs(d[0]*Math.Cosh(d[0]*d[1]+Math.Sin(d[2]))),
                (d)=>Math.Abs(Math.Tan(d[0]*d[1]+d[2])-Math.Sin(d[1]*d[2]+d[0])),
                (d)=>Math.Abs(Math.Pow(d[0]*d[0],d[1])-d[2]*Math.Exp(d[0])*Math.Log(d[2]*d[2]))
            };
}