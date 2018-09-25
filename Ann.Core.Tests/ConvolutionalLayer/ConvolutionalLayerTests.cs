using Ann.Activators;
using Ann.Core.Layers;
using Ann.Core.Tests.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Core.Tests
{
    [TestClass]
    public class ConvolutionalLayerTests
    {
        private readonly DoubleComparer _comparer;

        public ConvolutionalLayerTests()
        {
            _comparer = new DoubleComparer(3);
        }

        [TestMethod]
        [TestDataSource(0,2)]
        public void ForwardPassTest(int i)
        {
            var layer = CreateLayer(i);
            var actual = layer.PassForward(ConvolutionalLayerTestsData.ForwardPassInput[i]);
            var expected = ConvolutionalLayerTestsData.ForwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void BackwardPassTest(int i)
        {
            var layer = CreateLayer(i);
            layer.PassForward(ConvolutionalLayerTestsData.ForwardPassInput[i]);
            var actual = layer.PassBackward(ConvolutionalLayerTestsData.BackwardPassInput[i]);
            var expected = ConvolutionalLayerTestsData.BackwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        private Layer CreateLayer(int index)
        {
            var _layer = new ConvolutionLayer(
                4,
                3,
                new MessageShape(5, 3),
                new Flat(0.1),
                ActivatorType.Relu);

            _layer.SetWeights(ConvolutionalLayerTestsData.Weights[index]);

            return _layer;
        }
    }
}
