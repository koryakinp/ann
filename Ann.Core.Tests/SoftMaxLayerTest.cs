using Ann.Core.Layers;
using Ann.Core.WeightInitializers;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Core.Tests
{
    [TestClass]
    public class SoftMaxLayerTest
    {
        private SoftMaxLayer _layer { get; set; }
        private readonly List<double[]> _weights;
        private readonly double[] _input;
        private readonly double[] _error;
        private readonly double _lr;
        private readonly DoubleComparer _comparer;

        public SoftMaxLayerTest()
        {
            _comparer = new DoubleComparer(10);
            _weights = TestValues.Weights;
            _input = TestValues.Input;
            _error = TestValues.Error;
            _lr = TestValues.LR;
        }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new SoftMaxLayer(3, new MessageShape(1,4,1), new Flat(_lr));
            var queue = new Queue<double>(_weights.SelectMany(q => q));
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
            var temp = new double[_layer.Neurons.Count];
            for (int i = 0; i < _layer.Neurons.Count; i++)
            {
                temp[i] = Math.Exp(_layer.Neurons[i].Weights.Select((q, j) => q.Value * _input[j]).Sum());
            }

            for (int i = 0; i < _layer.Neurons.Count; i++)
            {
                expected[i] = temp[i] / temp.Sum();
            }

            Assert.AreEqual(1, expected.Sum(), Math.Pow(0.1, 10));
            CollectionAssert.AreEqual(expected, actual.ToSingle(), _comparer);
        }

        [TestMethod]
        public void PassBackwardTest()
        {
            _layer.PassForward(new Message(_input));

            var expected = new double[_input.Length];

            for (int i = 0; i < _layer.Neurons.Count; i++)
            {
                for (int j = 0; j < _layer.Neurons.Count; j++)
                {
                    var o1 = _layer.Neurons[i].Output;
                    var o2 = _layer.Neurons[j].Output;

                    var sd = i == j ? o1 * (1 - o1) : -o1 * o2;
                    _layer.Neurons[i].Delta += sd * _error[j];
                }
            }

            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < _layer.Neurons.Count; j++)
                {
                    var weight = _layer.Neurons[j].Weights[i].Value;
                    expected[i] += _layer.Neurons[j].Delta * weight;
                }
            }

            var actual = _layer.PassBackward(new Message(_error));
            CollectionAssert.AreEqual(expected, actual.ToSingle(), _comparer);
        }
    }
}
