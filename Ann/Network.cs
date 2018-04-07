using Ann.Activators;
using Ann.Layers;
using Ann.LossFunctions;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        private readonly List<Layer> _layers;
        private readonly LossFunction _lossFunction;
        private readonly LearningRateAnnealerType _lrat;
        private readonly int _numberOfInputs;
        private readonly int _numberOfOutputs;

        public Network(LossFunctionType lft, LearningRateAnnealerType lrat, int numberOfInputs, int numberOfOutputs)
        {
            _lossFunction = LossFunctionFactory.Produce(lft);
            _lrat = lrat;
            _numberOfInputs = numberOfInputs;
            _numberOfOutputs = numberOfOutputs;
            _layers = new List<Layer>();
        }

        public double TrainModel(double[] input, bool[] target)
        {
            var output = PassForward(input);
            PassBackward(ComputeCost(output, target));
            Learn(input);
            return ComputeError(output, target);
        }

        public double[] UseModel(double[] input)
        {
            return PassForward(input);
        }

        public void AddHiddenLayer(int numberOfNeurons, ActivatorType activatorType)
        {
            _layers.Add(new HiddenLayer(
                ActivatorFactory.Produce(activatorType),
                numberOfNeurons,
                _layers.Any() ? _layers.Last().Neurons.Count() : _numberOfInputs,
                _lrat,
                _layers.Any() ? _layers.Last().LayerIndex + 1 : 1));
        }

        public void AddOutputLayer()
        {
            _layers.Add(new OutputLayer(_numberOfOutputs, _layers.Last().Neurons.Count(), _lrat, _layers.Last().LayerIndex + 1));
        }

        public void FinalizeModel()
        {
            RandomizeWeights();
        }

        #region private methods
        private double[] ComputeCost(double[] output, bool[] target)
        {
            return _lossFunction.ComputeDeriviative(target, output);
        }

        private double ComputeError(double[] output, bool[] target)
        {
            return _lossFunction.ComputeLoss(target, output);
        }

        private void PassBackward(double[] value)
        {
            _layers
                .OrderByDescending(q => q.LayerIndex)
                .ForEach(q => value = q.PassBackward(value));
        }

        private void RandomizeWeights()
        {
            _layers.ForEach(q => q.RandomizeWeights());
        }

        private double[] PassForward(double[] value)
        {
            _layers.ForEach(q => value = q.PassForward(value));
            return value;
        }

        private void Learn(double[] input)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                var prevLayerOutput = _layers[i].LayerIndex == 1
                    ? input
                    : _layers[i - 1].Neurons.Select(q => q.Output).ToArray();

                _layers[i].UpdateWeights(prevLayerOutput);
                _layers[i].UpdateBiases();
            }
        }
        #endregion
    }
}
