using Ann.Activators;
using Ann.CostFunctions;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        private readonly List<Layer> _layers;
        private readonly ICostFunction _costFunction;
        private readonly int _numberOfInputs;

        public Network(CostFunctionType costFunctionType, int numberOfInputs)
        {
            _numberOfInputs = numberOfInputs;
            _costFunction = CostFunctionFactory.Produce(costFunctionType);
            _layers = new List<Layer>();
        }

        public double TrainModel(double[] input, double[] target)
        {
            var output = PassForward(input);
            PassBackward(ComputeCost(output, target));
            Learn();
            return ComputeError(output, target);
        }

        public double[] UseModel(double[] input)
        {
            return PassForward(input);
        }

        public void AddFullyConnectedLayer(int numberOfNeurons, ActivatorType activatorType, LearningRateAnnealerType lrat)
        {
            _layers.Add(new Layer(
                ActivatorFactory.Produce(activatorType),
                numberOfNeurons,
                _layers.Any() ? _layers.Last().Neurons.Count() : _numberOfInputs,
                lrat,
                _layers.Any() ? _layers.Last().LayerIndex + 1 : 1));
        }

        #region private methods
        private double[] ComputeCost(double[] output, double[] target)
        {
            return output
                .Select((q, i) => _costFunction.ComputeDeriviative(target[i], q))
                .ToArray();
        }

        private double ComputeError(double[] output, double[] target)
        {
            return output
                .Select((q, i) => _costFunction.ComputeValue(target[i], q))
                .Sum();
        }

        private void PassBackward(double[] value)
        {
            _layers
                .OrderByDescending(q => q.LayerIndex)
                .ForEach(q => value = q.PassBackward(value));
        }

        private double[] PassForward(double[] value)
        {
            _layers.ForEach(q => value = q.PassForward(value));
            return value;
        }

        private void Learn()
        {
            foreach (var layer in _layers)
            {
                layer.UpdateWeights();
                layer.UpdateBiases();
            }
        }
        #endregion
    }
}
