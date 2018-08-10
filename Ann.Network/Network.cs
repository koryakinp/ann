using Ann.Core;
using Ann.Core.LossFunctions;
using Ann.Core.WeightInitializers;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Network
{
    public class Network
    {
        internal readonly List<ILayer> _layers;
        private readonly LossFunction _lossFunction;
        internal readonly int _numberOfClasses;

        public Network(LossFunctionType lossFunctionType, int numberOfClasses)
        {
            _numberOfClasses = numberOfClasses;
            _lossFunction = LossFunctionFactory.Produce(lossFunctionType);
            _layers = new List<ILayer>();
        }

        public void AddLayer(LayerConfiguration layerInitializer)
        {
            _layers.Add(layerInitializer.CreateLayer(this));
        }

        public double TrainModel(Message input, bool[] labels)
        {
            var output = PassForward(input);
            Message error = new Message(_lossFunction.ComputeDeriviative(labels, output));

            PassBackward(error);
            Learn();

            return _lossFunction.ComputeLoss(labels, output);
        }

        public double[] UseModel(Message input)
        {
            return PassForward(input);
        }

        private double[] PassForward(Message input)
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                input = _layers[i].PassForward(input);
            }

            return input.ToSingle();
        }

        private void PassBackward(Message error)
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
