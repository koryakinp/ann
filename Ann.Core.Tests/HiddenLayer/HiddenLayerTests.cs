using Ann.Activators;
using Ann.Core.Tests.Utils;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Core.Tests.HiddenLayer
{
    [TestClass]
    public class HiddenLayerTests
    {
        private Layers.HiddenLayer _layer { get; set; }
        private readonly DoubleComparer _comparer;
        private readonly LogisticActivator _activator;
        private readonly int _numberOfNeurons;
        private readonly int _numberOfConnections;
        private readonly double _learningRate;

        public HiddenLayerTests()
        {
            _numberOfNeurons = 3;
            _numberOfConnections = 4;
            _learningRate = 0.1;
            _activator = new LogisticActivator();
            _comparer = new DoubleComparer(4);
        }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new Layers.HiddenLayer(
                _numberOfNeurons, 
                _activator, 
                new Flat(_learningRate), 
                new MessageShape(_numberOfConnections));
        }

        private void SeedWeights(int index)
        {
            var queue = new Queue<double>();
            HiddenLayerTestsData.Weights[index].ForEach(w => queue.Enqueue(w));
            Mock<IWeightInitializer> mock = new Mock<IWeightInitializer>();
            mock.Setup(q => q.GenerateRandom(It.IsAny<double>())).Returns(queue.Dequeue);
            _layer.RandomizeWeights(mock.Object);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void ForwardPassTest(int i)
        {
            SeedWeights(i);
            var actual = _layer.PassForward(HiddenLayerTestsData.ForwardPassInput[i]);
            var expected = HiddenLayerTestsData.ForwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void BackwardPassTest(int i)
        {
            SeedWeights(i);
            _layer.PassForward(HiddenLayerTestsData.ForwardPassInput[i]);
            var actual = _layer.PassBackward(HiddenLayerTestsData.BackwardPassInput[i]);
            var expected = HiddenLayerTestsData.BackwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        public void NumberOfNeuronsTest()
        {
            Assert.AreEqual(_layer.Neurons.Count, _numberOfNeurons);
            Assert.IsTrue(_layer.Neurons.All(q => q.Weights.Count() == _numberOfConnections));
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void ShouldUpdateWeights(int i)
        {
            SeedWeights(i);
            _layer.PassForward(HiddenLayerTestsData.ForwardPassInput[i]);
            _layer.PassBackward(HiddenLayerTestsData.BackwardPassInput[i]);
            _layer.UpdateWeights();
            var actual = _layer.Neurons.SelectMany(q => q.Weights.Select(w => w.Value)).ToArray();
            var expected = HiddenLayerTestsData.WeightsUpdated[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.HiddenLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfForwardInputMessageDimenionsNotValid()
        {
            _layer.PassForward(new double[4, 4]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.HiddenLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfForwardInputMessageDimenionsNotValid2()
        {
            _layer.PassForward(new double[10]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.HiddenLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfBackwardMessageDimenionsNotValid()
        {
            _layer.PassBackward(new double[4, 4]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.HiddenLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfBackwardMessageDimenionsNotValid2()
        {
            _layer.PassBackward(new double[10]);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void ShouldUpdateBiasTest(int i)
        {
            SeedWeights(i);
            _layer.PassForward(HiddenLayerTestsData.ForwardPassInput[i]);
            _layer.PassBackward(HiddenLayerTestsData.BackwardPassInput[i]);
            _layer.UpdateBiases();
            var actual = _layer.Neurons.Select(q => q.Bias.Value).ToArray();
            var expected = HiddenLayerTestsData.BiasesUpdated[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }
    }
}
