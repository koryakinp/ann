using Ann.Core;
using Ann.Core.Layers;
using Ann.Core.LossFunctions;
using Ann.Core.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        internal readonly List<Layer> _layers;
        private readonly LossFunction _lossFunction;
        internal readonly int _numberOfClasses;

        public Network(LossFunctionType lossFunctionType, int numberOfClasses)
        {
            _numberOfClasses = numberOfClasses;
            _lossFunction = LossFunctionFactory.Produce(lossFunctionType);
            _layers = new List<Layer>();
        }

        public double TrainModel(double[,] input, bool[] labels)
        {
            double[] output = PassForward(input).Cast<double>().ToArray();
            double[] error = _lossFunction.ComputeDeriviative(labels, output);

            PassBackward(error);
            Learn();

            return _lossFunction.ComputeLoss(labels, output);
        }

        public double[] UseModel(double[,] input)
        {
            return PassForward(input).Cast<double>().ToArray();
        }

        private Array PassForward(Array input)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                input = _layers[i].PassForward(input);
            }

            return input;
        }

        private void PassBackward(Array error)
        {
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                error = _layers[i].PassBackward(error);
            }
        }

        private void Learn()
        {
            foreach (var item in _layers.OfType<ILearnable>())
            {
                item.UpdateWeights();
                item.UpdateBiases();
            }
        }

        public void RandomizeWeights()
        {
            foreach (var layer in _layers.OfType<ILearnable>())
            {
                layer.RandomizeWeights(new WeightInitializer());
            }
        }
    }
}
