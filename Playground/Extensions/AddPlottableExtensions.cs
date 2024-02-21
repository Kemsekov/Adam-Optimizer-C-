using ScottPlot.Plottables;

namespace ScottPlot;
public static class AddPlottableExtensions
{
 
    public static ScottPlot.Color ToScatter(this System.Drawing.Color color){
        return Color.FromARGB((uint)color.ToArgb());
    }
}