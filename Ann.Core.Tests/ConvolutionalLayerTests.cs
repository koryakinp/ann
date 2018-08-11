using Ann.Activators;
using Ann.Core.Layers;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using System.Collections.Generic;

namespace Ann.Core.Tests
{
    [TestClass]
    class ConvolutionalLayerTests
    {
        public ConvolutionLayer _layer;
        private readonly int _numberOfKernels;
        private readonly int _kernelSize;
        private readonly int _numberOfFeatureMaps;
        private readonly int _featureMapSize;
        private readonly int _featureMapDepth;

        public ConvolutionalLayerTests()
        {
            _numberOfFeatureMaps = 2;
            _featureMapSize = 5;
            _numberOfKernels = 2;
            _kernelSize = 3;
            _featureMapDepth = 3;
        }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new ConvolutionLayer(
                _numberOfKernels,
                _kernelSize,
                new MessageShape(_featureMapSize, _featureMapSize, _featureMapDepth),
                new Flat(TestValues.LR), 
                ActivatorType.Relu);

            var queue = new Queue<double>();
            TestValues.Kernels.ForEach(q => queue.Enqueue(q));
            var mock = new Mock<IWeightInitializer>();
            mock.Setup(q => q.GenerateRandom(It.IsAny<double>()))
                .Returns(queue.Dequeue);
            _layer.RandomizeWeights(mock.Object);
        }

        public void ForwardPassTest()
        {
            var input = TestValues.MultiDimensionalInput;
            var actual = _layer.PassForward(input);
            int inputSize = TestValues.MultiDimensionalInput.GetLength(1);
            int outputSize = inputSize - _kernelSize + 1;
            var expected = new double[_numberOfKernels, outputSize, outputSize];

            expected.ForEach((i, j, k) => 
            {
            });
        }

    }
}
