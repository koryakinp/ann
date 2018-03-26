using Ann.Activators;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Tests
{
    [TestClass]
    public class NeuronTests
    {
        private Neuron neuron;

        [TestInitialize]
        public void SetUp()
        {
            Activator a = ActivatorFactory.Produce(ActivatorType.Sigmoid);
            neuron = new Neuron(a, 5, LearningRateAnnealerType.Adagrad);
        }

        [TestMethod]
        public void ShouldComputeOutput()
        {
            neuron.ComputeOutput(3);
            var expected = 0.952574126822;
            var actual = neuron.Output;

            Assert.AreEqual(expected, actual, 0.000001);
        }

        [TestMethod]
        public void ShouldComputeGradient()
        {
            neuron.ComputeOutput(3);
            neuron.ComputeDelta(3);
            var expected = 0.13552997919;
            var actual = neuron.Delta;

            Assert.AreEqual(expected, actual, 0.000001);
        }

        [TestMethod]
        public void ShouldUpdateWeights()
        {
            var weights = new double[] { 0.3, 0.2, -0.8, 1.3, 0.1 };

            neuron.SetWeights(weights);
            neuron.ComputeOutput(3);
            neuron.ComputeDelta(3);
            neuron.UpdateWeights();

            var expected = new double[] { 0, 0, 0, 0, 0 };

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], neuron.Weights[i].Value);
            }
        }
    }
}
