using Ann.Layers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Tests
{
    [TestClass]
    public class OutputLayerTests
    {
        internal OutputLayer _layer { get; set; }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new OutputLayer(3, 4, LearningRateAnnealerType.Adagrad, 1);
            _layer.Neurons[0].SetWeights(new double[] { 0.62, 1.54, -1.13, -0.36 });
            _layer.Neurons[1].SetWeights(new double[] { 1.87, -1.98, 1.76, -1.21 });
            _layer.Neurons[2].SetWeights(new double[] { -0.86, -0.20, 0.48, 0.73 });
        }

        [TestMethod]
        public void ForwardPassTest()
        {
            var output = _layer.PassForward(new double[] { 1.93, -1.10, -1.60, 1.55 });

            Assert.AreEqual(3, output.Length);

            output.ForEach(q => Assert.IsTrue(q >= 0));
            output.ForEach(q => Assert.IsTrue(q <= 1));

            var expected = new double[] { 0.38914, 0.54837, 0.06249 };

            output.ForEach((q,i) => Assert.AreEqual(expected[i], q, 0.00001));
        }


        [TestMethod]
        public void BackwardPassTest()
        {
            _layer.PassForward(new double[] { 1.93, -1.10, -1.60, 1.55 });
            var actual = _layer.PassBackward(new double[] { -1.5, 1.56, -1.99 });
            var expected = new double[] { 1.165974979, -2.494298372, 2.023643373, -0.804029415 };

            Assert.AreEqual(4, actual.Length);
            actual.ForEach((q, i) => Assert.AreEqual(expected[i], q, 0.00001));
        }
    }
}
