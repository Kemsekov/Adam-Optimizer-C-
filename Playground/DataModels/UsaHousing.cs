using MathNet.Numerics.LinearAlgebra.Single;

namespace Playground.DataModels;
public class UsaHousing
{
    public double AvgAreaIncome { get; set; }
    public double AvgHouseAge { get; set; }
    public double AvgNumberOfRooms { get; set; }
    public double AvgNumberOfBedrooms { get; set; }
    public double AreaPopulation { get; set; }
    public double Price { get; set; }
    public void NormalizeRecord(UsaHousing mean)
    {
        AvgAreaIncome       /= mean.AvgAreaIncome;
        AvgHouseAge         /= mean.AvgHouseAge;
        AvgNumberOfRooms    /= mean.AvgNumberOfRooms;
        AvgNumberOfBedrooms /= mean.AvgNumberOfBedrooms;
        AreaPopulation      /= mean.AreaPopulation;
        Price               /= mean.Price;
    }
    public UsaHousing RestoreRecord(UsaHousing mean)
    {
        var record = new UsaHousing(){
            AvgAreaIncome       = AvgAreaIncome       * mean.AvgAreaIncome,
            AvgHouseAge         = AvgHouseAge         * mean.AvgHouseAge,
            AvgNumberOfRooms    = AvgNumberOfRooms    * mean.AvgNumberOfRooms,
            AvgNumberOfBedrooms = AvgNumberOfBedrooms * mean.AvgNumberOfBedrooms,
            AreaPopulation      = AreaPopulation      * mean.AreaPopulation,
            Price               = Price               * mean.Price
        };
        return record;
    }
    public Vector ToInputVector(){
        var input = DenseVector.Create(5,0);
        input[0] = (float)AvgAreaIncome;
        input[1] = (float)AvgHouseAge;
        input[2] = (float)AvgNumberOfRooms;
        input[3] = (float)AvgNumberOfBedrooms;
        input[4] = (float)AreaPopulation;
        return input;
    }
    public override string ToString()
    {
        return 
            $"AvgAreaIncome\t\t{AvgAreaIncome}\n"+
            $"AvgHouseAge\t\t{AvgHouseAge}\n"+
            $"AvgNumberOfRooms\t{AvgNumberOfRooms}\n"+
            $"AvgNumberOfBedrooms\t{AvgNumberOfBedrooms}\n"+
            $"AreaPopulation\t\t{AreaPopulation}\n"+
            $"Price\t\t\t{Price}\n";
    }
}