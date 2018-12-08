using Ann.Core.Tests.Utils;
using Ann.Layers.Dense;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Ann.Tests.DenseLayer.DenseLayerTestData;

namespace Ann.Tests.DenseLayer
{
    [TestClass]
    public class DenseLayerTests
    {
        private DenseFullLayer _layer { get; set; }
        private readonly DoubleComparer _comparer;

        public DenseLayerTests()
        {
            _comparer = new DoubleComparer(5);
        }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new DenseFullLayer(new MessageShape(4), 3, true, new Flat(0.1));
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void ForwardPassTest(int i)
        {
            _layer.SetWeights(Weights[i]);
            _layer.SetBiases(Biases[i]);
            var actual = _layer.PassForward(DenseLayerForwardPassInput[i]);
            CollectionAssert.AreEqual(DenseLayerForwardPassOutput[i], actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void BackwardPassTest(int i)
        {
            _layer.SetWeights(Weights[i]);
            var actual = _layer.PassBackward(DenseLayerBackwardPassInput[i]);
            CollectionAssert.AreEqual(DenseLayerBackwardPassOutput[i], actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void UpdateTest(int i)
        {
            _layer.SetWeights(Weights[i]);
            _layer.SetBiases(Biases[i]);

            _layer.PassForward(DenseLayerForwardPassInput[i]);
            _layer.PassBackward(DenseLayerBackwardPassInput[i]);

            _layer.Update();
            var actualWeights = _layer.GetWeights();
            var actualBiases = _layer.GetBiases();

            CollectionAssert.AreEqual(BiasesUpdated[i], actualBiases, _comparer);
            CollectionAssert.AreEqual(WeightsUpdated[i], actualWeights, _comparer);
        }
    }
}
