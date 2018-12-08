using Ann.Persistence;
using Ann.Persistence.LayerConfig;
using Ann.Utils;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace Ann.Layers.Dense
{
    internal class DenseForwardLayer : BaseLayer, IForwardLayer
    {
        protected readonly int NumberOfNeurons;
        protected readonly Matrix<double> Weights;
        protected readonly Vector<double> Biases;
        protected readonly bool EnableBiases;

        public DenseForwardLayer(
            MessageShape inputMessageShape, 
            int numberOfNeurons,
            bool enableBiases) 
            : base(inputMessageShape, new MessageShape(numberOfNeurons))
        {
            EnableBiases = enableBiases;
            NumberOfNeurons = numberOfNeurons;
            Weights = Matrix.Build.Dense(inputMessageShape.Size, NumberOfNeurons);
            Biases = Vector.Build.Dense(NumberOfNeurons);
        }

        public LayerConfiguration GetConfiguration()
        {
            return new DenseLayerConfiguration(
                GetInputMessageShape(),
                EnableBiases,
                NumberOfNeurons,
                Weights.ToArray(),
                Biases.ToArray());
        }

        public Array PassForward(Array input)
        {
            var X = Matrix.Build.DenseOfRowArrays(input as double[]);

            return EnableBiases
                ? X.Multiply(Weights).Row(0).Add(Biases).ToArray()
                : X.Multiply(Weights).Row(0).ToArray();
        }

        public void SetBiases(double[] array)
        {
            Biases.SetValues(array);
        }

        public void SetWeights(Array array)
        {
            Weights.SetValues(array);
        }
    }
}
