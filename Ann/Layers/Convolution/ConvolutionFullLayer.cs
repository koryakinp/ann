using Ann.Misc;
using Ann.Utils;
using Gdo;
using MathNet.Numerics.Distributions;
using System;
using System.Linq;

namespace Ann.Layers.Convolution
{
    class ConvolutionFullLayer : ConvolutionForwardLayer, IFullLayer, ILearnable
    {
        private readonly double[,,] _cache;
        private readonly Optimizer[][,,] _weightOptimizers;
        private readonly Optimizer[] _biasOptimizers;

        public ConvolutionFullLayer(
            MessageShape inputMessageShape,
            int kernelSize,
            int numberOfkernels,
            Optimizer optimizer)
            : base(inputMessageShape, kernelSize, numberOfkernels)
        {
            _cache = new double[inputMessageShape.Depth, inputMessageShape.Size, inputMessageShape.Size];
            _weightOptimizers = Helper.InitializeKernelOptimizers(
                inputMessageShape.Depth, numberOfkernels, kernelSize, optimizer);
            _biasOptimizers = Helper.InitializeBiasOptimizers(numberOfkernels, optimizer);
        }

        public Array PassBackward(Array error)
        {
            var gradients = error as double[,,];
            ComputeFilterGradients(gradients);
            ComputeBiasGradient(gradients);
            return ComputeInputGradients(gradients);
        }

        public new Array PassForward(Array input)
        {
            _cache.UpdateForEach<double>((q, idx) => (double)input.GetValue(idx));
            return base.PassForward(input);
        }

        public void RandomizeWeights(double stddev)
        {
            var dist = new Normal(0, stddev);
            Kernels
                .ForEach((q, kernel) => 
                    q.UpdateForEach<double>((w, idx) => dist.Sample()));

            Kernels
                .ForEach((q, kernel) => q.ForEach((w, i, j, k) => 
                    _weightOptimizers[kernel][i, j, k].SetValue(w)));
        }

        public void Update()
        {
            _biasOptimizers.ForEach((q, i) => Biases[i] = q.Update());
            Kernels.ForEach((kernel, kernelIndex) =>
            {
                kernel.UpdateForEach<double>((q, idx) => 
                    ((Optimizer)_weightOptimizers[kernelIndex].GetValue(idx)).Update());
            });
        }

        private void ComputeBiasGradient(double[,,] gradients)
        {
            for (int i = 0; i < gradients.GetLength(0); i++)
            {
                _biasOptimizers[i].SetGradient(gradients.GetChannel(i).Cast<double>().Sum());
            }
        }

        private void ComputeFilterGradients(double[,,] gradients)
        {
            _weightOptimizers
                .AsParallel()
                .Select((q, i) => new { Kernel = q, Index = i })
                .ForAll(q =>
                {
                    var dEdO = gradients.GetChannel(q.Index);
                    for (int j = 0; j < _cache.GetLength(0); j++)
                    {
                        var dedf = MatrixHelper.Convolution(_cache.GetChannel(j), dEdO);
                        q.Kernel
                            .GetChannel(j)
                            .ForEach((w, ii, jj) => w.SetGradient(dedf[ii, jj]));
                    }
                });
        }

        private double[,,] ComputeInputGradients(double[,,] input)
        {
            var transposed = Kernels
                .Transpose()
                .Select(q => q.Rotate())
                .ToArray();

            var padded = input.Pad(KernelSize - 1);

            return MatrixHelper.Convolution(transposed, padded);
        }


        public double[][,,] GetWeights()
        {
            return Kernels.Select(q => q).ToArray();
        }

        public double[] GetBiases()
        {
            return Biases.ToArray();
        }

        public new void SetBiases(double[] biases)
        {
            base.SetBiases(biases);
            biases.ForEach((q, i) => _biasOptimizers[i].SetValue(q));
        }

        public new void SetWeights(Array weights)
        {
            base.SetWeights(weights);
            (weights as double[][,,])
                .ForEach((q, kernel) => q.ForEach((w, i, j, k) => 
                    _weightOptimizers[kernel][i, j, k].SetValue(w)));

        }
    }
}
