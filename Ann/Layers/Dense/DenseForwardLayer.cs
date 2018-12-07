using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Layers.Dense
{
    class DenseForwardLayer : BaseLayer, IForwardLayer
    {
        protected readonly int NumberOfNeurons;
        protected readonly Matrix<double> Weights;
        protected readonly Vector<double> Biases;

        public DenseForwardLayer(
            MessageShape inputMessageShape, 
            int numberOfNeurons) 
            : base(inputMessageShape, new MessageShape(numberOfNeurons))
        {
            NumberOfNeurons = numberOfNeurons;
            Weights = Matrix.Build.Dense(inputMessageShape.Size, NumberOfNeurons);
            Biases = Vector.Build.Dense(NumberOfNeurons);
        }

        public Array PassForward(Array input)
        {
            var X = Matrix.Build.DenseOfRowArrays(input as double[]);
            return X.Multiply(Weights).Row(0).Add(Biases).ToArray();
        }
    }
}
