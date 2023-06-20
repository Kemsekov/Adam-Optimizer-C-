using System.Diagnostics;
using GradientDescentSharp.Utils.Kernels;
using ILGPU;
using ILGPU.Runtime;
using MathNet.Numerics.LinearAlgebra.Single;

var watch = new Stopwatch();
using var context = Context.Create(b=>{
    b.Optimize(OptimizationLevel.Release);
    b.Default();
});
using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);
var provider = new LinearAlgebraProvider(accelerator);
var size = 10000;
using var gvec1 = accelerator.Allocate1D<float>(size);
using var gvec2 = accelerator.Allocate1D<float>(size);
using var g1Result = accelerator.Allocate1D<float>(size);
using var g2Result = accelerator.Allocate1D<float>(size);
using var rofl = accelerator.Allocate1D<float>(size);

var vec1 = DenseVector.Create(size, x => Random.Shared.NextSingle());
var vec2 = DenseVector.Create(size, x => Random.Shared.NextSingle());
gvec1.CopyFromCPU(vec1.Values);
gvec2.CopyFromCPU(vec2.Values);

var mat = DenseMatrix.Create(size,size,(i,j)=>Random.Shared.NextSingle());
using var gmat = accelerator.Allocate2DDenseY<float>((size,size));
gmat.CopyFromCPU(mat.ToArray());
LinearAlgebraProvider.CompileAllKernels();

for(int i = 0;i<10;i++){
    provider.MatrixVectorMul(gmat,rofl,rofl); // 17 avg
    provider.MatrixVectorMul(rofl,gmat,rofl); // 17 avg
    provider.AddVectors(rofl,rofl,1,rofl); // 17 avg
}
 
var res = mat*vec1+vec2*mat; //340 avg
provider.MatrixVectorMul(gmat,gvec1,g1Result); // 17 avg
watch.Start();
provider.MatrixVectorMul(gvec2,gmat,g2Result); // 17 avg
System.Console.WriteLine("Done in " + watch.ElapsedMilliseconds);
provider.AddVectors(g1Result,g2Result,1,rofl); //8 avg

var copy = new float[size];
rofl.CopyToCPU(copy);
var diff = 0.0f;
for(int i = 0;i<size;i++){
    diff+=Math.Abs(copy[i]-res[i]);
}
System.Console.WriteLine("total error "+diff);