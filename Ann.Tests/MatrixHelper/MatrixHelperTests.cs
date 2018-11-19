using Ann.Core.Tests.Utils;
using Ann.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Ann.MatrixHelper;

namespace Ann.Core.Tests.MatrixHelper
{
    [TestClass]
    public class MatrixHelperTests
    {
        private readonly DoubleComparer _comparer;

        public MatrixHelperTests()
        {
            _comparer = new DoubleComparer(4);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void RotateTest(int i)
        {
            var actual = MatrixHelperTestData.RotateInput[i].Rotate();
            var expected = MatrixHelperTestData.RotateOutput[i];

            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void TransposeTest(int i)
        {
            var actual = MatrixHelperTestData.TransposeInput[i].Transpose();
            var expected = MatrixHelperTestData.TransposeOutput[i];

            Assert.AreEqual(actual.Length, expected.Length);
            for (int j = 0; j < actual.Length; j++)
            {
                CollectionAssert.AreEqual(expected[j], actual[j]); 
            }
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void PadTest(int i)
        {
            var actual = MatrixHelperTestData.PadInput[i].Pad(2);
            var expected = MatrixHelperTestData.PadOutput[i];

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void MaxPoolEvenTest(int i)
        {
            var actual = MaxPool(MatrixHelperTestData.MaxPoolEvenInput[i], 2);
            var expectedValues = MatrixHelperTestData.MaxPoolEvenOutput[i];
            var expectedCache = MatrixHelperTestData.MaxPoolEvenCacheOutput[i];

            CollectionAssert.AreEqual(expectedValues, actual.Values, _comparer);
            CollectionAssert.AreEqual(expectedCache, actual.Cache);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void MaxPoolOddTest(int i)
        {
            var actual = MaxPool(MatrixHelperTestData.MaxPoolOddInput[i], 2);
            var expectedValues = MatrixHelperTestData.MaxPoolOddOutput[i];
            var expectedCache = MatrixHelperTestData.MaxPoolOddCacheOutput[i];

            CollectionAssert.AreEqual(expectedValues, actual.Values, _comparer);
            CollectionAssert.AreEqual(expectedCache, actual.Cache);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void ConvolutionTest(int i)
        {
            var X = MatrixHelperTestData.ConvolutionInput[i];
            var W = MatrixHelperTestData.ConvolutionWeights[i];

            var actual = Convolution(W, X);

            CollectionAssert.AreEqual(MatrixHelperTestData.ConvolutionOutput[i], actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void ConvolutionSingleLayerTest(int i)
        {
            var actual = MatrixHelperTestData
                .ConvolutionSingleLayerInput[i]
                .Convolution(MatrixHelperTestData.ConvolutionSingleLayerWeights[i]);

            CollectionAssert.AreEqual(MatrixHelperTestData.ConvolutionSingleLayerOutput[i], actual, _comparer);

        }
    }
}
