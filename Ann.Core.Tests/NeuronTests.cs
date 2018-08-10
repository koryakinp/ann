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
    public class NeuronTests
    {
        private Neuron _neuron { get; set; }
        private readonly double[] _weights;

        public NeuronTests()
        {
            _weights = new double[5] { 1, 2, 3, 4, 5 };
        }

        [TestInitialize]
        public void Initialize()
        {
            _neuron = new Neuron(Enumerable.Range(1, 5).Select(q => new Flat(0.1)).ToArray(), new Flat(0.1));
        }

        [TestMethod]
        public void ShouldSetWeights()
        {
            var mock = new Mock<IWeightInitializer>();
            var queue = new Queue<double>(_weights);
            mock.Setup(q => q.GenerateRandom(It.IsAny<double>()))
                .Returns(queue.Dequeue);

            _neuron.RandomizeWeights(mock.Object);
            var actual = _neuron.Weights.Select(q => q.Value).ToArray();
            CollectionAssert.AreEqual(_weights, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.CanNotUpdateWeights)]
        public void ShouldThrowUpdateWeights()
        {
            _neuron.UpdateWeights(new double[6]);
        }

        [TestMethod]
        public void ShouldUpdateWeights()
        {
            var mock = new Mock<IWeightInitializer>();
            var queue = new Queue<double>(_weights);
            mock.Setup(q => q.GenerateRandom(It.IsAny<double>()))
                .Returns(queue.Dequeue);

            _neuron.RandomizeWeights(mock.Object);
            _neuron.Delta = 1;
            _neuron.UpdateWeights(new double[5] { 2, 2, 2, 2, 2 });

            var actual = _neuron.Weights.Select(q => q.Value).ToArray();
            CollectionAssert.AreEqual(new double[5] { 0.8, 1.8, 2.8, 3.8, 4.8 }, actual);
        }

        [TestMethod]
        public void ShouldUpdateBias()
        {
            _neuron.Bias.SetValue(5);
            _neuron.Delta = 1;
            _neuron.UpdateBias();
            Assert.AreEqual(4.9, _neuron.Bias.Value);
        }


    }
}
