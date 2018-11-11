using Gdo;
using System.Linq;
using Activator = Ann.Activators.Activator;
using System;
using MathNet.Numerics.LinearAlgebra.Double;
using Ann.Utils;

namespace Ann.Core.Layers
{
    public class HiddenLayer : NeuronLayer
    {
        private readonly Activator _activator;

        public HiddenLayer(
            int numberOfNeurons,
            Activator activator,
            Optimizer optimizer,
            MessageShape inputMessageShape) 
            : base(numberOfNeurons, inputMessageShape, optimizer)
        {
            _activator = activator;
        }

        public override Array PassForward(Array input)
        {
            PrevLayerOutput = input;

            var X = Matrix.Build.Dense(1, input.Length, input.OfType<double>().ToArray());
            var W = GetWeightMatrix();
            var B = new DenseVector(Neurons.Select(q => q.Bias.Value).ToArray());

            var res = X.Multiply(W).Row(0).Add(B).Map(q => _activator.CalculateValue(q)).ToArray();
            Neurons.ForEach((q, i) => q.Output = res[i]);

            return Neurons.Select(q => q.Output).ToArray();
        }

        public override Array PassBackward(Array error)
        {
            var W = GetWeightMatrix();
            Neurons.ForEach((q, i) => q.Delta = (double)error.GetValue(i) * _activator.CalculateDeriviative(q.Output));
            var dEdX = Neurons.Select(q => q.Delta).ToArray();
            var dEdO = Matrix.Build.Dense(1, dEdX.Length, dEdX);
            return dEdO.TransposeAndMultiply(W).Row(0).ToArray();
        }
    }
}
