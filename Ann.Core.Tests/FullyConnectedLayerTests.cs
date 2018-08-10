using Ann.Activators;
using Ann.Core.Layers;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Core.Tests
{
    [TestClass]
    public class FullyConnectedLayerTests
    {
        private HiddenLayer _layer { get; set; }

        private readonly List<double[]> _weights;
        private readonly double[] _input;
        private readonly double[] _error;
        private readonly LogisticActivator _activator;
        private readonly double _lr;
        private readonly DoubleComparer _comparer;

        public FullyConnectedLayerTests()
        {
            _comparer = new DoubleComparer(10);
            _comparer = new DoubleComparer(10);
            _weights = TestValues.Weights;
            _input = TestValues.Input;
            _error = TestValues.Error;
            _lr = TestValues.LR;
            _activator = new LogisticActivator();
        }


        [TestInitialize]
        public void Initialize()
        {
            _layer = new HiddenLayer(3, _activator, new Flat(_lr), new MessageShape(1,4,1));
            var queue = new Queue<double>();

            _weights.ForEach(q => q.ForEach(w => queue.Enqueue(w)));

            Mock<IWeightInitializer> mock = new Mock<IWeightInitializer>();
            mock.Setup(q => q.GenerateRandom(It.IsAny<double>())).Returns(queue.Dequeue);
            _layer.RandomizeWeights(mock.Object);
        }

        [TestMethod]
        public void RandomizeWeightsTest()
        {
            for (int i = 0; i < _layer.Neurons.Count; i++)
            {
                var actual = _layer.Neurons[i].Weights.Select(q => q.Value).ToArray();
                CollectionAssert.AreEqual(actual, _weights[i], _comparer);
            }
        }

        [TestMethod]
        public void PassForwardTest()
        {
            var actual = _layer.PassForward(new Message(_input));
            var expected = new double[_layer.Neurons.Count];

            for (int i = 0; i < _layer.Neurons.Count; i++)
            {
                var sum = _layer.Neurons[i].Weights.Select((q, j) => q.Value * _input[j]).Sum();
                expected[i] = 1 / (1 + Math.Exp(-sum));
            }

            CollectionAssert.AreEqual(expected, actual.ToSingle(), _comparer);
        }

        [TestMethod]
        public void PassBackwardTest()
        {
            _layer.PassForward(new Message(_input));
            var actual = _layer.PassBackward(new Message(_error));
            var expected = new double[_input.Length];

            for (int i = 0; i < _layer.Neurons.Count; i++)
            {
                var output = _layer.Neurons[i].Output;
                var error = _error[i];
                _layer.Neurons[i].Delta = output * (1 - output) * error;
            }

            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < _layer.Neurons.Count; j++)
                {
                    var weight = _layer.Neurons[j].Weights[i].Value;
                    expected[i] += _layer.Neurons[j].Delta * weight;
                }
            }

            CollectionAssert.AreEqual(expected, actual.ToSingle(), _comparer);
        }

        [TestMethod]
        public void UpdateWeightsTest()
        {
            _layer.PassForward(new Message(_input));
            _layer.Neurons.ForEach((q, i) => _layer.Neurons[i].Delta = 2);
            _layer.UpdateWeights();
            var actual = new List<double[]>();
            _layer.Neurons.ForEach(q => actual.Add(q.Weights.Select(w => w.Value).ToArray()));

            var expected = new List<double[]>();
            for (int i = 0; i < _weights.Count; i++)
            {
                expected.Add(_weights[i].Select((q, j) => q - _layer.Neurons[i].Delta * _input[j] * _lr).ToArray());
            }

            for (int i = 0; i < expected.Count; i++)
            {
                CollectionAssert.AreEqual(expected[i], actual[i], _comparer);
            }
        }

        [TestMethod]
        public void UpdateBiasTest()
        {
            _layer.Neurons.ForEach((q, i) => _layer.Neurons[i].Delta = 2);
            _layer.Neurons.ForEach((q, i) => _layer.Neurons[i].Bias.SetValue(3));
            var expected = _layer.Neurons.Select(q => q.Bias.Value - q.Delta * _lr).ToArray();
            _layer.UpdateBiases();
            var actual = _layer.Neurons.Select(q => q.Bias.Value).ToArray();

            CollectionAssert.AreEqual(actual, expected, _comparer);

        }
    }
}
