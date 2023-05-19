using ScottPlot.Plottables;

namespace ScottPlot;
public static class AddPlottableExtensions
{
    public static Scatter Point(this AddPlottable plt,double x,double y, Color? color = null)
    {
        return plt.Scatter(new[]{x,x},new[]{y,y},color);
    }
    public static ScottPlot.Color ToScatter(this System.Drawing.Color color){
        return Color.FromARGB((uint)color.ToArgb());
    }
}