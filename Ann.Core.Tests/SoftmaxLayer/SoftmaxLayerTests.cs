using Ann.Core.Tests.Utils;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using System.Collections.Generic;

namespace Ann.Core.Tests.SoftmaxLayer
{
    [TestClass]
    public class SoftmaxLayerTests
    {
        private Layers.SoftMaxLayer _layer { get; set; }
        private readonly double _learningRate;
        public readonly int _numberOfNeurons;
        public readonly int _numberOfNeuronsInPreviouseLayer;
        private readonly DoubleComparer _comparer;

        public SoftmaxLayerTests()
        {
            _learningRate = 0.1;
            _numberOfNeurons = 3;
            _numberOfNeuronsInPreviouseLayer = 4;
            _comparer = new DoubleComparer(4);
        }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new Layers.SoftMaxLayer(_numberOfNeurons, 
                new MessageShape(_numberOfNeuronsInPreviouseLayer), 
                new Flat(_learningRate));
        }

        private void SeedWeights(int index)
        {
            var queue = new Queue<double>();
            SoftmaxLayerTestsData.Weights[index].ForEach(w => queue.Enqueue(w));
            Mock<IWeightInitializer> mock = new Mock<IWeightInitializer>();
            mock.Setup(q => q.GenerateRandom(It.IsAny<double>())).Returns(queue.Dequeue);
            _layer.RandomizeWeights(mock.Object);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void PassForwardTest(int i)
        {
            SeedWeights(i);
            var actual = _layer.PassForward(SoftmaxLayerTestsData.ForwardPassInput[i]);
            var expected = SoftmaxLayerTestsData.ForwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void PassBackwardTest(int i)
        {
            SeedWeights(i);
            _layer.PassForward(SoftmaxLayerTestsData.ForwardPassInput[i]);
            var actual = _layer.PassBackward(SoftmaxLayerTestsData.BackwardPassInput[i]);
            var expected = SoftmaxLayerTestsData.BackwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }
    }
}
