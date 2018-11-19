using Ann.Activators;
using Ann.Core.Tests.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ActivationLayer = Ann.Layers.ActivationLayer;

namespace Ann.Core.Tests.ActivationLayer
{
    [TestClass]
    public class ActivationLayerTests
    {
        public Layers.ActivationLayer _layer;
        private readonly DoubleComparer _comparer;

        public ActivationLayerTests()
        {
            _layer = new Layers.ActivationLayer(new MessageShape(5, 3), ActivatorType.Sigmoid);
            _comparer = new DoubleComparer(4);
        }

        [TestMethod]
        [TestDataSource(0,2)]
        public void ForwardPassTest(int i)
        {
            var actual = _layer.PassForward(ActivationLayerTestsData.ForwardPassInput[i]);
            var expected = ActivationLayerTestsData.ForwardPassOutput[i];

            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void BackwardPassTest(int i)
        {
            _layer.PassForward(ActivationLayerTestsData.ForwardPassInput[i]);
            var actual = _layer.PassBackward(ActivationLayerTestsData.BackwardPassInput[i]);
            var expected = ActivationLayerTestsData.BackwardPassOutput[i];

            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

    }
}
