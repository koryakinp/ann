using Ann.Core.Tests.Utils;
using Ann.Layers.Convolution;
using Ann.Tests;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using static Ann.Core.Tests.ConvolutionalLayer.ConvolutionalLayerTestsData;

namespace Ann.Core.Tests.ConvolutionalLayer
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
        [TestDataSource(0,3)]
        public void SetWeightsTest(int index)
        {
            var layer = CreateLayer();
            layer.SetWeights(Weights[index]);
            layer
                .GetWeights()
                .ForEach((kernel, kk) => kernel
                    .ForEach((q, i, j, k) =>
                    {
                        Assert.AreEqual(q, Weights[index][kk][i, j, k]);
                    }));
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.CanNotSetWeights)]
        public void SetWeightsShouldThrowIfShapeIsInvalidTest1()
        {
            var layer = CreateLayer();

            layer.SetWeights(new double[10]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.CanNotSetWeights)]
        public void SetWeightsShouldThrowIfShapeIsInvalidTest2()
        {
            var layer = CreateLayer();

            layer.SetWeights(new double[5][,,] 
            {
                new double[3,3,3],
                new double[3,3,3],
                new double[3,3,3],
                new double[3,3,3],
                new double[3,3,3],
            });
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.CanNotSetWeights)]
        public void SetWeightsShouldThrowIfShapeIsInvalidTest3()
        {
            var layer = CreateLayer();

            layer.SetWeights(new double[4][,,]
            {
                new double[3,3,3],
                new double[3,3,3],
                new double[3,4,3],
                new double[3,3,3]
            });
        }

        [TestMethod]
        [TestDataSource(0,3)]
        public void ForwardPassTest(int i)
        {
            var layer = CreateLayer();
            layer.SetWeights(Weights[i]);
            var actual = layer.PassForward(ForwardPassInput[i]);
            var expected = ForwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void ForwardPassWithBiasTest(int i)
        {
            var layer = CreateLayer();
            layer.SetWeights(Weights[i]);
            layer.SetBiases(new double[4] { 1, 1, 1, 1 });
            var actual = layer.PassForward(ForwardPassInput[i]);
            var expected = (double[,,])ForwardPassOutput[i].Clone();
            expected.UpdateForEach<double>(q => q + 1);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }


        [TestMethod]
        [TestDataSource(0,3)]
        public void BackwardPassTest(int i)
        {
            var layer = CreateLayer();
            layer.SetWeights(Weights[i]);
            layer.PassForward(ForwardPassInput[i]);
            var actual = layer.PassBackward(BackwardPassInput[i]);
            var expected = BackwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void BackwardPassWeightGradientsTest(int i)
        {
            var layer = CreateLayer();
            layer.SetWeights(Weights[i]);
            layer.PassForward(ForwardPassInput[i]);
            layer.PassBackward(BackwardPassInput[i]);

            /*
            for (int k = 0; k < layer..Length; k++)
            {
                var actual = layer._kernels[k].Gradients;
                var expected = WeightGradients[i][k];
                CollectionAssert.AreEqual(expected, actual, _comparer);
            }
            */
        }

        private ConvolutionFullLayer CreateLayer()
        {
            return new ConvolutionFullLayer(new MessageShape(5, 3), 3, 4, new Flat(0.1));
        }
    }
}
