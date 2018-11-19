using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace Ann.Core.Tests
{
    [TestClass]
    public class NeuronTests
    {
        private Neuron _neuron { get; set; }

        [TestInitialize]
        public void Initialize()
        {
            _neuron = new Neuron(Enumerable.Range(1, 5).Select(q => new Flat(0.1)).ToArray(), new Flat(0.1));
        }

        [TestMethod]
        public void ShouldSetWeights()
        {
            _neuron.RandomizeWeights(0.1);
            _neuron.Weights.ForEach(q => Assert.AreNotEqual(0, q.Value));
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
            _neuron.RandomizeWeights(0.1);
            _neuron.Delta = 1;
            var weights = _neuron.Weights.Select(q => q.Value - 2 * 0.1).ToArray();
            _neuron.UpdateWeights(new double[5] { 2, 2, 2, 2, 2 });
            var actual = _neuron.Weights.Select(q => q.Value).ToArray();
            CollectionAssert.AreEqual(weights, actual);
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
