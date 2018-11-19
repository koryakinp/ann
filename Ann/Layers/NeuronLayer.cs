using System;
using System.Collections.Generic;
using System.Linq;
using Ann.Utils;
using Gdo;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Ann.Layers
{
    public abstract class NeuronLayer : Layer, ILearnable
    {
        public readonly IReadOnlyList<Neuron> Neurons;
        protected Array PrevLayerOutput;

        public NeuronLayer(
            int numberOfNeurons,
            MessageShape inputMessageShape,
            Optimizer optimizer) : base(inputMessageShape, new MessageShape(numberOfNeurons))
        {
            List<Neuron> nList = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Optimizer[] weights = new Optimizer[InputMessageShape.Size];
                for (int j = 0; j < InputMessageShape.Size; j++)
                {
                    weights[j] = optimizer.Clone() as Optimizer;
                }
                nList.Add(new Neuron(weights, optimizer.Clone() as Optimizer));
            }

            Neurons = new List<Neuron>(nList);
        }

        public void UpdateWeights()
        {
            Neurons.ForEach(q => q.UpdateWeights(PrevLayerOutput.Cast<double>().ToArray()));
        }

        public void UpdateBiases()
        {
            Neurons.ForEach(q => q.UpdateBias());
        }

        public void RandomizeWeights(double stddev)
        {
            Neurons.ForEach(q => q.RandomizeWeights(stddev));
        }

        public void SetWeights(Array weights)
        {
            if(Neurons.Count() != weights.Length)
            {
                throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
            }
            else if(!(weights is double[][]))
            {
                throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
            }

            var temp = weights as double[][];
            for (int i = 0; i < Neurons.Count; i++)
            {
                var neuron = Neurons[i];
                if (neuron.Weights.Count() != temp[i].Length)
                {
                    throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
                }
            }

            Neurons.ForEach((q, i) =>
            {
                if (q.Weights.Count() != temp[i].Length)
                {
                    throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
                }
            });

            for (int i = 0; i < Neurons.Count; i++)
            {
                var neuron = Neurons[i];
                for (int j = 0; j < Neurons[i].Weights.Length; j++)
                {
                    var weight = neuron.Weights[j];
                    weight.SetValue(temp[i][j]);
                }
            }
        }

        public override void ValidateForwardInput(Array input)
        {
            base.ValidateForwardInput(input);

            if(input.Rank != 1)
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
            else if(input.Length != InputMessageShape.Size)
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
        }

        public override void ValidateBackwardInput(Array input)
        {
            base.ValidateForwardInput(input);

            if (input.Rank > 1)
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
            else if (input.Length != Neurons.Count)
            {
                throw new Exception(Consts.CommonLayerMessages.MessageDimenionsInvalid);
            }
        }

        protected Matrix<double> GetWeightMatrix()
        {
            return Matrix.Build.DenseOfColumnArrays(
                Neurons.Select(q => q.Weights.Select(w => w.Value).ToArray()).ToArray());
        }

        internal double[][] GetWeights()
        {
            return Neurons.Select(q => q.Weights.Select(w => w.Value).ToArray()).ToArray();
        }

        internal double[] GetBiases()
        {
            return Neurons.Select(q => q.Bias.Value).ToArray();
        }

        public void SetBiases(Array biases)
        {
            Neurons.ForEach((q, i) => q.Bias.SetValue((double)biases.GetValue(i)));
        }
    }
}
