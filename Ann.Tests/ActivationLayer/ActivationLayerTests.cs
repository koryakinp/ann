using Ann.Activators;
using Ann.Core.Tests.Utils;
using Ann.Persistence.LayerConfig;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Tests.ActivationLayer
{
    [TestClass]
    public class ActivationLayerTests
    {
        public readonly DoubleComparer _comparer;
        private Layers.ActivationLayer _layer;

        public ActivationLayerTests()
        {
            _comparer = new DoubleComparer(4);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void ForwardPassTest(int i)
        {
            _layer = Create3D();

            var expected = ActivationLayerTestData.ForwardPass3DOutput[i];
            var actual = _layer.PassForward(ActivationLayerTestData.ForwardPass3DInput[i]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void BackwardPassTest(int i)
        {
            _layer = Create3D();

            var expected = ActivationLayerTestData.BackwardPass3DOutput[i];
            _layer.PassForward(ActivationLayerTestData.ForwardPass3DInput[i]);
            var actual = _layer.PassBackward(ActivationLayerTestData.BackwardPass3DInput[i]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void ForwardPass1DTest(int i)
        {
            _layer = Create1D();

            var expected = ActivationLayerTestData.ForwardPass1DOutput[i];
            var actual = _layer.PassForward(ActivationLayerTestData.ForwardPass1DInput[i]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void BackwardPass1DTest(int i)
        {
            _layer = Create1D();

            var expected = ActivationLayerTestData.BackwardPass1DOutput[i];
            _layer.PassForward(ActivationLayerTestData.ForwardPass1DInput[i]);
            var actual = _layer.PassBackward(ActivationLayerTestData.BackwardPass1DInput[i]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        private Layers.ActivationLayer Create3D()
        {
            var config = new ActivationLayerConfiguration(new MessageShape(5, 3), ActivatorType.Sigmoid);
            return new Layers.ActivationLayer(config);
        }

        private Layers.ActivationLayer Create1D()
        {
            var config = new ActivationLayerConfiguration(new MessageShape(10), ActivatorType.Sigmoid);
            return new Layers.ActivationLayer(config);
        }
    }
}
