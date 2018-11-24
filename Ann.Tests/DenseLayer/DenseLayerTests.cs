using Ann.Core.Tests.Utils;
using Ann.Persistence.LayerConfig;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Ann.Tests.DenseLayer.DenseLayerTestData;

namespace Ann.Tests.DenseLayer
{
    [TestClass]
    public class DenseLayerTests
    {
        private Layers.DenseLayer _layer { get; set; }
        private readonly DoubleComparer _comparer;

        public DenseLayerTests()
        {
            _comparer = new DoubleComparer(5);
        }

        [TestInitialize]
        public void Initialize()
        {
            var config = new DenseLayerConfiguration(new MessageShape(4), new Flat(0.1), 3);
            _layer = new Layers.DenseLayer(config);
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
        public void UpdateWeightsTest(int i)
        {
            _layer.SetWeights(Weights[i]);
            _layer.SetBiases(Biases[i]);

            _layer.PassForward(DenseLayerForwardPassInput[i]);
            _layer.PassBackward(DenseLayerBackwardPassInput[i]);

            _layer.UpdateWeights();
            var actual = _layer.GetWeights();

            CollectionAssert.AreEqual(WeightsUpdated[i], actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 2)]
        public void UpdateBiasesTest(int i)
        {
            _layer.SetWeights(Weights[i]);
            _layer.SetBiases(Biases[i]);

            _layer.PassForward(DenseLayerForwardPassInput[i]);
            _layer.PassBackward(DenseLayerBackwardPassInput[i]);

            _layer.UpdateBiases();
            var actual = _layer.GetBiases();

            CollectionAssert.AreEqual(BiasesUpdated[i], actual, _comparer);
        }
    }
}
