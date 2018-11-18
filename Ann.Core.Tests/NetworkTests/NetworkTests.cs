using Ann.Activators;
using Ann.Core.Layers;
using Ann.Core.LossFunctions;
using Ann.Core.Tests.Utils;
using Ann.Core.WeightInitializers;
using Ann.Utils;
using Gdo.Optimizers;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ann.Core.Tests.NetworkTests
{
    [TestClass]
    public class NetworkTests
    {
        public Network network;
        private readonly DoubleComparer _comparer;

        public NetworkTests()
        {
            _comparer = new DoubleComparer(5);
        }

        [TestInitialize]
        public void Initialize()
        {
            network = new Network(LossFunctionType.CrossEntropy, 3);
        }

        [TestMethod]
        public void CNNTest()
        {
            network.AddInputLayer(7, 1);
            network.AddConvolutionLayer(new Flat(0.1), 2, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddConvolutionLayer(new Flat(0.1), 3, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddFlattenLayer();
            network.AddHiddenLayer(5, ActivatorType.Relu, new Flat(0.1));
            network.AddSoftMaxLayer(new Flat(0.1));

            network.SetWeights(0, NetworkTestsData.Conv1Weights);
            network.SetWeights(1, NetworkTestsData.Conv2Weights);
            network.SetWeights(2, NetworkTestsData.Dense1Weights);
            network.SetWeights(3, NetworkTestsData.Dense2Weights);

            network.TrainModel(NetworkTestsData.Input, NetworkTestsData.Labels);
            var learnableLayers = network._layers.ToArray();
            var conv1layer = learnableLayers[1] as ConvolutionLayer;
            var conv2layer = learnableLayers[4] as ConvolutionLayer;
            var dense1layer = learnableLayers[8] as NeuronLayer;
            var dense2layer = learnableLayers[9] as NeuronLayer;

            var w1 = conv1layer.GetWeights() as double[][,,];
            var w2 = conv2layer.GetWeights() as double[][,,];
            var w3 = dense1layer.GetWeights() as double[][];
            var w4 = dense2layer.GetWeights() as double[][];

            var b1 = conv1layer.GetBiases() as double[];
            var b2 = conv2layer.GetBiases() as double[];
            var b3 = dense1layer.GetBiases() as double[];

            CollectionAssert.AreEqual(b1, NetworkTestsData.Conv1BiasesUpdated, _comparer);
            CollectionAssert.AreEqual(b2, NetworkTestsData.Conv2BiasesUpdated, _comparer);
            CollectionAssert.AreEqual(b3, NetworkTestsData.Dense1BiasesUpdated, _comparer);

            w1.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Conv1WeightsUpdated[i], _comparer);
            });

            w2.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Conv2WeightsUpdated[i], _comparer);
            });

            w3.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Dense1WeightsUpdated[i], _comparer);
            });

            w4.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Dense2WeightsUpdated[i], _comparer);
            });
        }
    }
}
