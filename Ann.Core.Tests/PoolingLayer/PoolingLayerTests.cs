using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Core.Tests.PoolingLayer
{
    [TestClass]
    public class PoolingLayerTests
    {
        private readonly int _stride;

        public PoolingLayerTests()
        {
            _stride = 2;
        }

        public Layers.PoolingLayer GetLayer(int size)
        {
            return new Layers.PoolingLayer(_stride, new MessageShape(size, 3));
        }

        [TestMethod]
        [TestDataSource(0,6)]
        public void PassForwardEvenTest(int i)
        {
            Layers.PoolingLayer layer = GetLayer(6);
            var input = MatrixHelper.MatrixHelperTestData.MaxPoolEvenInput[i];
            var expected = MatrixHelper.MatrixHelperTestData.MaxPoolEvenOutput[i];
            var actual = layer.PassForward(input);
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [TestDataSource(0,6)]
        public void PassForwardOddTest(int i)
        {
            Layers.PoolingLayer layer = GetLayer(5);
            var input = MatrixHelper.MatrixHelperTestData.MaxPoolOddInput[i];
            var expected = MatrixHelper.MatrixHelperTestData.MaxPoolOddOutput[i];
            var actual = layer.PassForward(input);
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [TestDataSource(0,6)]
        public void PassBackwardEvenTest(int i)
        {
            Layers.PoolingLayer layer = GetLayer(6);

            layer.PassForward(MatrixHelper.MatrixHelperTestData.MaxPoolEvenInput[i]);
            var actual = layer.PassBackward(MatrixHelper.MatrixHelperTestData.ReverseMaxPoolInput[i]);
            var expected = MatrixHelper.MatrixHelperTestData.ReverseMaxPoolEvenOutput[i];

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [TestDataSource(0,6)]
        public void PassBackwardOddTest(int i)
        {
            Layers.PoolingLayer layer = GetLayer(5);

            layer.PassForward(MatrixHelper.MatrixHelperTestData.MaxPoolOddInput[i]);
            var actual = layer.PassBackward(MatrixHelper.MatrixHelperTestData.ReverseMaxPoolInput[i]);
            var expected = MatrixHelper.MatrixHelperTestData.ReverseMaxPoolOddOutput[i];

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
