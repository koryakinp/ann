using Ann.Layers;
using Ann.LossFunctions;
using Ann.Utils;
using Gdo;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        internal readonly List<IFullLayer> _layers;
        private readonly LossFunction _lossFunction;
        internal readonly int _numberOfClasses;
        internal readonly Optimizer _optimizer;

        public Network(LossFunctionType lossFunctionType, Optimizer optimizer, int numberOfClasses)
        {
            _numberOfClasses = numberOfClasses;
            _lossFunction = LossFunctionFactory.Produce(lossFunctionType);
            _layers = new List<IFullLayer>();
            _optimizer = optimizer;
        }

        public void TrainModel(Array input, bool[] labels)
        {
            double[] output = PassForward(input).Cast<double>().ToArray();
            double[] error = _lossFunction.ComputeDeriviative(labels, output);

            PassBackward(error);
            Learn();
        }

        private Array PassForward(Array input)
        {
            _layers.ForEach(q => input = q.PassForward(input));
            return input;
        }

        private void PassBackward(Array error)
        {
            _layers.Reverse<IFullLayer>()
                .ForEach(q => error = q.PassBackward(error));
        }

        private void Learn()
        {
            _layers.OfType<ILearnable>()
                .ForEach(q => q.Update());
        }

        public void RandomizeWeights(double stddev)
        {
            _layers.OfType<ILearnable>()
                .ForEach(q => q.RandomizeWeights(stddev));
        }

        internal void SetWeights(int layerIndex, Array weights)
        {
            _layers.OfType<ILearnable>()
                .ToArray()[layerIndex].SetWeights(weights);
        }

        internal void SetBiases(int layerIndex, double[] biases)
        {
            _layers.OfType<ILearnable>()
                .ToArray()[layerIndex].SetBiases(biases);
        }

        public Model BuildModel()
        {
            var lc = _layers
                .OfType<IForwardLayer>()
                .Select(q => q.GetConfiguration())
                .ToList();

            return new Model(lc);
        }
    }
}
