using Ann.Activators;
using Ann.Core.Tests.Utils;
using Ann.Layers;
using Ann.LossFunctions;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;

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
            var lr = 0.1;

            network.AddInputLayer(7, 1);
            network.AddConvolutionLayer(new Flat(lr), 2, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddConvolutionLayer(new Flat(lr), 3, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddFlattenLayer();
            network.AddDenseLayer(5, true, new Flat(lr));
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(3, false, new Flat(lr));
            network.AddSoftMaxLayer();

            network.SetWeights(0, NetworkTestsData.Conv1Weights);
            network.SetWeights(1, NetworkTestsData.Conv2Weights);
            network.SetWeights(2, NetworkTestsData.Dense1Weights);
            network.SetWeights(3, NetworkTestsData.Dense2Weights);

            network.TrainModel(NetworkTestsData.Input, NetworkTestsData.Labels[0]);

            var learnableLayers = network._layers.ToArray();
            var conv1layer = learnableLayers[1] as ConvolutionLayer;
            var conv2layer = learnableLayers[4] as ConvolutionLayer;
            var dense1layer = learnableLayers[8] as DenseLayer;
            var dense2layer = learnableLayers[10] as DenseLayer;

            var w1 = conv1layer.GetWeights();
            var w2 = conv2layer.GetWeights();
            var w3 = dense1layer.GetWeights();
            var w4 = dense2layer.GetWeights();

            var b1 = conv1layer.GetBiases();
            var b2 = conv2layer.GetBiases();
            var b3 = dense1layer.GetBiases();

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

            CollectionAssert.AreEqual(w3, NetworkTestsData.Dense1WeightsUpdated, _comparer);
            CollectionAssert.AreEqual(w4, NetworkTestsData.Dense2WeightsUpdated, _comparer);
        }


        [TestMethod]
        public void SaveToFileTest()
        {
            network.AddInputLayer(7, 1);
            network.AddConvolutionLayer(new Flat(0.1), 2, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddConvolutionLayer(new Flat(0.1), 3, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddFlattenLayer();
            network.AddDenseLayer(5, true, new Flat(0.1));
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(3, false, new Flat(0.1));
            network.AddSoftMaxLayer();

            network.SetWeights(0, NetworkTestsData.Conv1Weights);
            network.SetWeights(1, NetworkTestsData.Conv2Weights);
            network.SetWeights(2, NetworkTestsData.Dense1Weights);
            network.SetWeights(3, NetworkTestsData.Dense2Weights);

            network.TrainModel(NetworkTestsData.Input, NetworkTestsData.Labels[0]);

            network.SaveModel("network.json");
        }

        [TestMethod]
        public void LoadFromFile()
        {
            network.AddInputLayer(7, 1);
            network.AddConvolutionLayer(new Flat(0.1), 2, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddConvolutionLayer(new Flat(0.1), 3, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddFlattenLayer();
            network.AddDenseLayer(5, true, new Flat(0.1));
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(3, false, new Flat(0.1));
            network.AddSoftMaxLayer();

            network.SetWeights(0, NetworkTestsData.Conv1Weights);
            network.SetWeights(1, NetworkTestsData.Conv2Weights);
            network.SetWeights(2, NetworkTestsData.Dense1Weights);
            network.SetWeights(3, NetworkTestsData.Dense2Weights);

            network.TrainModel(NetworkTestsData.Input, NetworkTestsData.Labels[0]);
            network.SaveModel("network.json");

            var network2 = new Network("network.json");
            var learnableLayers = network2._layers.ToArray();
            var conv1layer = learnableLayers[1] as ConvolutionLayer;
            var conv2layer = learnableLayers[4] as ConvolutionLayer;
            var dense1layer = learnableLayers[8] as DenseLayer;
            var dense2layer = learnableLayers[10] as DenseLayer;

            var w1 = conv1layer.GetWeights();
            var w2 = conv2layer.GetWeights();
            var w3 = dense1layer.GetWeights();
            var w4 = dense2layer.GetWeights();

            var b1 = conv1layer.GetBiases();
            var b2 = conv2layer.GetBiases();
            var b3 = dense1layer.GetBiases();

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

            CollectionAssert.AreEqual(w3, NetworkTestsData.Dense1WeightsUpdated, _comparer);
            CollectionAssert.AreEqual(w4, NetworkTestsData.Dense2WeightsUpdated, _comparer);
        }
    }
}
