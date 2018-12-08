using Ann.Activators;
using Ann.Core.Tests.Utils;
using Ann.Layers.Activation;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Tests.ActivationLayer
{
    [TestClass]
    public class ActivationLayerTests
    {
        public readonly DoubleComparer _comparer;

        public ActivationLayerTests()
        {
            _comparer = new DoubleComparer(4);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void ForwardPassTest(int i)
        {
            var _layer = Create3D();

            var expected = ActivationLayerTestData.ForwardPass3DOutput[i];
            var actual = _layer.PassForward(ActivationLayerTestData.ForwardPass3DInput[i].Clone() as double[,,]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void BackwardPassTest(int i)
        {
            var _layer = Create3D();

            var expected = ActivationLayerTestData.BackwardPass3DOutput[i];
            _layer.PassForward(ActivationLayerTestData.ForwardPass3DInput[i].Clone() as double[,,]);
            var actual = _layer.PassBackward(ActivationLayerTestData.BackwardPass3DInput[i].Clone() as double[,,]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void ForwardPass1DTest(int i)
        {
            var _layer = Create1D();

            var expected = ActivationLayerTestData.ForwardPass1DOutput[i];
            var actual = _layer.PassForward(ActivationLayerTestData.ForwardPass1DInput[i].Clone() as double[]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void BackwardPass1DTest(int i)
        {
            var _layer = Create1D();

            var expected = ActivationLayerTestData.BackwardPass1DOutput[i];
            _layer.PassForward(ActivationLayerTestData.ForwardPass1DInput[i].Clone() as double[]);
            var actual = _layer.PassBackward(ActivationLayerTestData.BackwardPass1DInput[i].Clone() as double[]);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        private ActivationFullLayer Create3D()
        {
            return new ActivationFullLayer(ActivatorType.Sigmoid, new MessageShape(5, 3));
        }

        private ActivationFullLayer Create1D()
        {
            return new ActivationFullLayer(ActivatorType.Sigmoid, new MessageShape(10));
        }
    }
}
