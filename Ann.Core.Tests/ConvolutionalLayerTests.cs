using Ann.Activators;
using Ann.Core.Layers;
using Ann.Core.Tests.Utils;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using System;
using System.Collections.Generic;

namespace Ann.Core.Tests
{
    [TestClass]
    public class ConvolutionalLayerTests
    {
        private readonly int _numberOfKernels;
        private readonly int _kernelSize;
        private readonly int _featureMapSize;
        private readonly int _featureMapDepth;
        private readonly DoubleComparer _comparer;

        public ConvolutionalLayerTests()
        {
            _featureMapSize = 5;
            _numberOfKernels = 4;
            _kernelSize = 3;
            _featureMapDepth = 3;
            _comparer = new DoubleComparer(3);
        }

        [TestMethod]
        public void ForwardPassTest()
        {
            for (int i = 0; i < 3; i++)
            {
                var layer = CreateLayer(ConvolutionalLayerValues.Weights[i]);
                var actual = layer.PassForward(ConvolutionalLayerValues.ForwardPassInput[i]);
                var expected = ConvolutionalLayerValues.ForwardPassOutput[i];
                CollectionAssert.AreEqual(expected, actual, _comparer);
            }
        }

        [TestMethod]
        public void BackwardPassTest()
        {
            for (int i = 0; i < 3; i++)
            {
                var layer = CreateLayer(ConvolutionalLayerValues.Weights[i]);
                layer.PassForward(ConvolutionalLayerValues.ForwardPassInput[i]);
                var actual = layer.PassBackward(ConvolutionalLayerValues.BackwardPassInput[i]);
                var expected = ConvolutionalLayerValues.BackwardPassOutput[i];
                CollectionAssert.AreEqual(expected, actual, _comparer);
            }
        }

        private Layer CreateLayer(double[] weights)
        {
            var _layer = new ConvolutionLayer(
                _numberOfKernels,
                _kernelSize,
                new MessageShape(_featureMapSize, _featureMapDepth),
                new Flat(TestValues.LR),
                ActivatorType.Relu);

            var queue = new Queue<double>(weights);
            TestValues.Kernels.ForEach(q => queue.Enqueue(q));
            var mock = new Mock<IWeightInitializer>();
            mock.Setup(q => q.GenerateRandom(It.IsAny<double>()))
                .Returns(queue.Dequeue);
            _layer.RandomizeWeights(mock.Object);
            return _layer;
        }
    }
}
