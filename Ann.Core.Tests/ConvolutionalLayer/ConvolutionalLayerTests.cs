using Ann.Activators;
using Ann.Core.Layers;
using Ann.Core.Tests.Utils;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;

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
            var layer = new ConvolutionLayer(4,3,
                new MessageShape(5, 3),
                new Flat(0.1));

            layer.SetWeights(ConvolutionalLayerTestsData.Weights[index]);
            for (int kernel = 0; kernel < layer._kernels.Length; kernel++)
            {
                layer._kernels[kernel].Weights.ForEach((q, i, j, k) =>
                {
                    Assert.AreEqual(q.Value, 
                        ConvolutionalLayerTestsData.Weights[index][kernel][i, j, k]);
                });
            }
        }

        [TestMethod]
        [TestDataSource(0,3)]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.CanNotSetWeights)]
        public void SetWeightsShouldThrowIfShapeIsInvalidTest1(int index)
        {
            var layer = new ConvolutionLayer(4, 3,
                new MessageShape(5, 3),
                new Flat(0.1));

            layer.SetWeights(new double[10]);
        }

        [TestMethod]
        [TestDataSource(0,3)]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.CanNotSetWeights)]
        public void SetWeightsShouldThrowIfShapeIsInvalidTest2(int index)
        {
            var layer = new ConvolutionLayer(4, 3,
                new MessageShape(5, 3),
                new Flat(0.1));

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
        [TestDataSource(0,3)]
        [ExpectedException(typeof(Exception), Consts.CommonLayerMessages.CanNotSetWeights)]
        public void SetWeightsShouldThrowIfShapeIsInvalidTest3(int index)
        {
            var layer = new ConvolutionLayer(4, 3,
                new MessageShape(5, 3),
                new Flat(0.1));

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
            var layer = CreateLayer(i);
            var actual = layer.PassForward(ConvolutionalLayerTestsData.ForwardPassInput[i]);
            var expected = ConvolutionalLayerTestsData.ForwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void ForwardPassWithBiasTest(int i)
        {
            var layer = CreateLayer(i);
            layer._kernels.ForEach(q => q.Bias.SetValue(1));
            var actual = layer.PassForward(ConvolutionalLayerTestsData.ForwardPassInput[i]);
            var expected = (double[,,])ConvolutionalLayerTestsData.ForwardPassOutput[i].Clone();
            expected.UpdateForEach<double>(q => q + 1);
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }


        [TestMethod]
        [TestDataSource(0,3)]
        public void BackwardPassTest(int i)
        {
            var layer = CreateLayer(i);
            layer.PassForward(ConvolutionalLayerTestsData.ForwardPassInput[i]);
            var actual = layer.PassBackward(ConvolutionalLayerTestsData.BackwardPassInput[i]);
            var expected = ConvolutionalLayerTestsData.BackwardPassOutput[i];
            CollectionAssert.AreEqual(expected, actual, _comparer);
        }

        [TestMethod]
        [TestDataSource(0, 3)]
        public void BackwardPassWeightGradientsTest(int i)
        {
            var layer = CreateLayer(i);
            layer.PassForward(ConvolutionalLayerTestsData.ForwardPassInput[i]);
            layer.PassBackward(ConvolutionalLayerTestsData.BackwardPassInput[i]);

            for (int k = 0; k < layer._kernels.Length; k++)
            {
                var actual = layer._kernels[k].Gradients;
                var expected = ConvolutionalLayerTestsData.WeightGradients[i][k];
                CollectionAssert.AreEqual(expected, actual, _comparer);
            }
        }

        private ConvolutionLayer CreateLayer(int index)
        {
            var _layer = new ConvolutionLayer(4,3, new MessageShape(5, 3), new Flat(0.1));

            _layer.SetWeights(ConvolutionalLayerTestsData.Weights[index]);

            return _layer;
        }
    }
}
