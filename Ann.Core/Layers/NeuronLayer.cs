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

        public void RandomizeWeights(IWeightInitializer initializer)
        {
            Neurons.ForEach(q => q.RandomizeWeights(initializer));
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

            Neurons.ForEach((q, i) =>
            {
                var temp = weights.GetValue(i) as double[];
                if (q.Weights.Count() != temp.Length)
                {
                    throw new Exception(Consts.CommonLayerMessages.CanNotSetWeights);
                }
            });

            Neurons.ForEach((q, i) =>
            {
                q.Weights.ForEach((w, j) => w.SetValue((double)weights.GetValue(i, j)));
            });
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
    }
}
