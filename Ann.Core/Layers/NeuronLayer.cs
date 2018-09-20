using System;
using System.Collections.Generic;
using System.Linq;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo;

namespace Ann.Core.Layers
{
    public abstract class NeuronLayer : Layer, ILearnable
    {
        public readonly IReadOnlyList<Neuron> Neurons;
        protected Array PrevLayerOutput;

        public NeuronLayer(
            int numberOfNeurons,
            MessageShape inputMessageShape,
            Optimizer optimizer) : base(inputMessageShape)
        {
            List<Neuron> nList = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Optimizer[] weights = new Optimizer[InputMessageShape.Height];
                for (int j = 0; j < InputMessageShape.Height; j++)
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

        public void RandomizeWeights(IWeightInitializer initializer)
        {
            Neurons.ForEach(q => q.RandomizeWeights(initializer));
        }

        public override MessageShape GetOutputMessageShape()
        {
            return new MessageShape(1, Neurons.Count, 1);
        }

        protected void ValidateForwardInput(Array input)
        {
            if(input.Rank != 1)
            {
                throw new Exception(Consts.HiddenLayerMessages.MessageDimenionsInvalid);
            }
            else if(input.Length != InputMessageShape.Height)
            {
                throw new Exception(Consts.HiddenLayerMessages.MessageDimenionsInvalid);
            }
        }

        protected void ValidateBackwardInput(Array input)
        {
            if (input.Rank > 1)
            {
                throw new Exception(Consts.HiddenLayerMessages.MessageDimenionsInvalid);
            }
            else if (input.Length != Neurons.Count)
            {
                throw new Exception(Consts.HiddenLayerMessages.MessageDimenionsInvalid);
            }
        }
    }
}
