using Ann.Core.Tests.Utils;
using Ann.Persistence.LayerConfig;
using Ann.Tests;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Core.Tests.SoftmaxLayer
{
    [TestClass]
    public class SoftmaxLayerTests
    {
        private Layers.SoftMaxLayer _layer { get; set; }
        public readonly int _numberOfNeurons;
        private readonly DoubleComparer _comparer;

        public SoftmaxLayerTests()
        {
            _numberOfNeurons = 3;
            _comparer = new DoubleComparer(4);
        }

        [TestInitialize]
        public void Initialize()
        {
            var config = new SoftmaxLayerConfiguration(new MessageShape(_numberOfNeurons));
            _layer = new Layers.SoftMaxLayer(config);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void PassForwardTest(int i)
        {
            var actual = _layer.PassForward(SoftmaxLayerTestsData.ForwardPassInput[i]);
            var expected = SoftmaxLayerTestsData.ForwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void PassBackwardTest(int i)
        {
            _layer.PassForward(SoftmaxLayerTestsData.ForwardPassInput[i]);
            var actual = _layer.PassBackward(SoftmaxLayerTestsData.BackwardPassInput[i]);
            var expected = SoftmaxLayerTestsData.BackwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }
    }
}
