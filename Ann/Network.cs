using Ann.Layers;
using Ann.LossFunctions;
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
                item.Update();
            }
        }

        public void RandomizeWeights(double stddev)
        {
            foreach (var layer in _layers.OfType<ILearnable>())
            {
                layer.RandomizeWeights(stddev);
            }
        }

        internal void SetWeights(int layerIndex, Array weights)
        {
            _layers.OfType<ILearnable>().ToArray()[layerIndex].SetWeights(weights);
        }

        internal void SetBiases(int layerIndex, double[] biases)
        {
            _layers.OfType<ILearnable>().ToArray()[layerIndex].SetBiases(biases);
        }

        public Model BuildModel()
        {
            //var lc = _layers.Select(q => q.GetLayerConfiguration()).ToList();
            //return new Model(lc);
            throw new NotImplementedException();
        }
    }
}
