using Ann.Activators;
using Ann.Core.Tests.Utils;
using Ann.Tests;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace Ann.Core.Tests.HiddenLayer
{
    [TestClass]
    public class HiddenLayerTests
    {
        private Layers.HiddenLayer _layer { get; set; }
        private readonly DoubleComparer _comparer;
        private readonly ActivatorType _activatorType;
        private readonly int _numberOfNeurons;
        private readonly int _numberOfConnections;
        private readonly double _learningRate;

        public HiddenLayerTests()
        {
            _numberOfNeurons = 3;
            _numberOfConnections = 4;
            _learningRate = 0.1;
            _activatorType = ActivatorType.Sigmoid;
            _comparer = new DoubleComparer(4);
        }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new Layers.HiddenLayer(
                _numberOfNeurons, 
                _activatorType, 
                new Flat(_learningRate), 
                new MessageShape(_numberOfConnections));
        }

        private void SeedWeights(int index)
        {
            _layer.SetWeights(HiddenLayerTestsData.Weights[index]);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void ForwardPassTest(int i)
        {
            SeedWeights(i);
            var actual = _layer.PassForward(HiddenLayerTestsData.ForwardPassInput[i]);
            var expected = HiddenLayerTestsData.ForwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0,3)]
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
        [TestDataSource(0,3)]
        public void SetWeightsTest(int i)
        {
            _layer.SetWeights(HiddenLayerTestsData.Weights[i]);
            for (int n = 0; n < _layer.Neurons.Count; n++)
            {
                for (int w = 0; w < _layer.Neurons[n].Weights.Count(); w++)
                {
                    Assert.AreEqual(
                        _layer.Neurons[n].Weights[w].Value,
                        HiddenLayerTestsData.Weights[i][n][w]);
                }
            }
        }

        [TestMethod]
        [TestDataSource(0,3)]
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
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfForwardInputMessageDimenionsNotValid()
        {
            _layer.ValidateForwardInput(new double[4, 4]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfForwardInputMessageDimenionsNotValid2()
        {
            _layer.ValidateForwardInput(new double[10]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfBackwardMessageDimenionsNotValid()
        {
            _layer.ValidateBackwardInput(new double[4, 4]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.MessageDimenionsInvalid)]
        public void ShouldThrowIfBackwardMessageDimenionsNotValid2()
        {
            _layer.ValidateBackwardInput(new double[10]);
        }

        [TestMethod]
        [TestDataSource(0,3)]
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
