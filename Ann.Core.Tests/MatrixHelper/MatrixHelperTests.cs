using Ann.Core.Tests.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Ann.Core.MatrixHelper;

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
        public void RotateTest()
        {
            for (int i = 0; i < MatrixHelperTestData.RotateInput.Length; i++)
            {
                var actual = Rotate(MatrixHelperTestData.RotateInput[i]);
                var expected = MatrixHelperTestData.RotateOutput[i];

                CollectionAssert.AreEqual(expected, actual);
            }
        }

        [TestMethod]
        public void TransposeTest()
        {
            for (int i = 0; i < MatrixHelperTestData.TransposeInput.Length; i++)
            {
                var actual = Transpose(MatrixHelperTestData.TransposeInput[i]);
                var expected = MatrixHelperTestData.TransposeOutput[i];

                Assert.AreEqual(actual.Length, expected.Length);
                for (int j = 0; j < actual.Length; j++)
                {
                    CollectionAssert.AreEqual(expected[j], actual[j]); 
                }
            }
        }

        [TestMethod]
        public void PadTest()
        {
            for (int i = 0; i < MatrixHelperTestData.PadInput.Length; i++)
            {
                var actual = Pad(MatrixHelperTestData.PadInput[i], 2);
                var expected = MatrixHelperTestData.PadOutput[i];

                CollectionAssert.AreEqual(expected, actual);
            }
        }

        [TestMethod]
        public void MaxPoolEvenTest()
        {
            for (int i = 0; i < MatrixHelperTestData.MaxPoolEvenInput.Length; i++)
            {
                var actual = MaxPool(MatrixHelperTestData.MaxPoolEvenInput[i], 2);
                var expectedValues = MatrixHelperTestData.MaxPoolEvenOutput[i];
                var expectedCache = MatrixHelperTestData.MaxPoolEvenCacheOutput[i];

                CollectionAssert.AreEqual(expectedValues, actual.Values, _comparer);
                CollectionAssert.AreEqual(expectedCache, actual.Cache);
            }
        }

        [TestMethod]
        public void MaxPoolOddTest()
        {
            for (int i = 0; i < MatrixHelperTestData.MaxPoolOddInput.Length; i++)
            {
                var actual = MaxPool(MatrixHelperTestData.MaxPoolOddInput[i], 2);
                var expectedValues = MatrixHelperTestData.MaxPoolOddOutput[i];
                var expectedCache = MatrixHelperTestData.MaxPoolOddCacheOutput[i];

                CollectionAssert.AreEqual(expectedValues, actual.Values, _comparer);
                CollectionAssert.AreEqual(expectedCache, actual.Cache);
            }
        }
    }
}
