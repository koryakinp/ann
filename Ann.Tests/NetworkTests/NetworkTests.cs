using Ann.Activators;
using Ann.Core.Tests.Utils;
using Ann.Layers;
using Ann.LossFunctions;
using Ann.Mnist;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using System.IO;

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

            w3.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Dense1WeightsUpdated[i], _comparer);
            });

            w4.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Dense2WeightsUpdated[i], _comparer);
            });
        }

        [TestMethod]
        public void CompareNetworksTest()
        {
            var network = new Network(LossFunctionType.CrossEntropy, 10);

            var lr = 0.001;

            network.AddInputLayer(28, 1, true);
            network.AddConvolutionLayer(Optimizers.Flat(lr), 16, 5, true);
            network.AddActivationLayer(ActivatorType.Relu, true);
            network.AddPoolingLayer(2, true);
            network.AddConvolutionLayer(Optimizers.Flat(lr), 32, 5, true);
            network.AddActivationLayer(ActivatorType.Relu, true);
            network.AddPoolingLayer(2, true);
            network.AddFlattenLayer(true);
            network.AddHiddenLayer(1024, ActivatorType.Relu, Optimizers.Flat(lr), true);
            network.AddSoftMaxLayer(Optimizers.Flat(lr), true);

            network.AddInputLayer(28, 1, false);
            network.AddConvolutionLayer(Optimizers.Flat(lr), 16, 5, false);
            network.AddActivationLayer(ActivatorType.Relu, false);
            network.AddPoolingLayer(2, false);
            network.AddConvolutionLayer(Optimizers.Flat(lr), 32, 5, false);
            network.AddActivationLayer(ActivatorType.Relu, false);
            network.AddPoolingLayer(2, false);
            network.AddFlattenLayer(false);
            network.AddDenseLayer(1024, true, Optimizers.Flat(lr), false);
            network.AddActivationLayer(ActivatorType.Relu, false);
            network.AddDenseLayer(10, false, Optimizers.Flat(lr), false);
            network.AddSoftMaxLayer2(false);

            network.SetWeights(0, ReadConvWeightsJson("before/conv_1_weights.json"), true);
            network.SetBiases(0, ReadDenseWeightsJson<double[]>("before/conv_1_biases.json"), true);
            network.SetWeights(1, ReadConvWeightsJson("before/conv_2_weights.json"), true);
            network.SetBiases(1, ReadDenseWeightsJson<double[]>("before/conv_2_biases.json"), true);
            network.SetWeights(2, MapWeights(ReadDenseWeightsJson<double[,]>("before/dense_1_weights.json")), true);
            network.SetBiases(2, ReadDenseWeightsJson<double[]>("before/dense_1_biases.json"), true);
            network.SetWeights(3, MapWeights(ReadDenseWeightsJson<double[,]>("before/dense_2_weights.json")), true);

            network.SetWeights(0, ReadConvWeightsJson("before/conv_1_weights.json"), false);
            network.SetBiases(0, ReadDenseWeightsJson<double[]>("before/conv_1_biases.json"), false);
            network.SetWeights(1, ReadConvWeightsJson("before/conv_2_weights.json"), false);
            network.SetBiases(1, ReadDenseWeightsJson<double[]>("before/conv_2_biases.json"), false);
            network.SetWeights(2, ReadDenseWeightsJson<double[,]>("before/dense_1_weights.json"), false);
            network.SetBiases(2, ReadDenseWeightsJson<double[]>("before/dense_1_biases.json"), false);
            network.SetWeights(3, ReadDenseWeightsJson<double[,]>("before/dense_2_weights.json"), false);

            foreach (var image in MnistReader.ReadTrainingData(10000))
            {
                var target = Helper.CreateTarget(image.Label);
                network.TrainModel(Helper.Create3DInput(image.Data), target);
            }
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
            network.AddHiddenLayer(5, ActivatorType.Relu, new Flat(0.1));
            network.AddSoftMaxLayer(new Flat(0.1));

            network.SetWeights(0, NetworkTestsData.Conv1Weights);
            network.SetWeights(1, NetworkTestsData.Conv2Weights);
            network.SetWeights(2, NetworkTestsData.Dense1Weights);
            network.SetWeights(3, NetworkTestsData.Dense2Weights);

            network.TrainModel(NetworkTestsData.Input, NetworkTestsData.Labels);

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
            network.AddHiddenLayer(5, ActivatorType.Relu, new Flat(0.1));
            network.AddSoftMaxLayer(new Flat(0.1));

            network.SetWeights(0, NetworkTestsData.Conv1Weights);
            network.SetWeights(1, NetworkTestsData.Conv2Weights);
            network.SetWeights(2, NetworkTestsData.Dense1Weights);
            network.SetWeights(3, NetworkTestsData.Dense2Weights);

            network.TrainModel(NetworkTestsData.Input, NetworkTestsData.Labels);
            network.SaveModel("network.json");

            var network2 = new Network("network.json");
            var learnableLayers = network2._layers.ToArray();
            var conv1layer = learnableLayers[1] as ConvolutionLayer;
            var conv2layer = learnableLayers[4] as ConvolutionLayer;
            var dense1layer = learnableLayers[8] as NeuronLayer;
            var dense2layer = learnableLayers[9] as NeuronLayer;

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

            w3.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Dense1WeightsUpdated[i], _comparer);
            });

            w4.ForEach((q, i) =>
            {
                CollectionAssert.AreEqual(q, NetworkTestsData.Dense2WeightsUpdated[i], _comparer);
            });
        }

        private T ReadDenseWeightsJson<T>(string file)
        {
            var json1 = File.ReadAllText($"NetworkTests/{file}");
            return JsonConvert.DeserializeObject<T>(json1);
        }

        private double[][,,] ReadConvWeightsJson(string file)
        {
            var json1 = File.ReadAllText($"NetworkTests/{file}");
            var w1 = JsonConvert.DeserializeObject<double[,,,]>(json1);

            var output = new double[w1.GetLength(3)][,,];

            for (int kernel = 0; kernel < w1.GetLength(3); kernel++)
            {
                var arr = new double[w1.GetLength(2), w1.GetLength(1), w1.GetLength(0)];

                for (int d = 0; d < w1.GetLength(2); d++)
                {
                    for (int i = 0; i < w1.GetLength(1); i++)
                    {
                        for (int j = 0; j < w1.GetLength(0); j++)
                        {
                            arr[d, i, j] = w1[i, j, d, kernel];
                        }
                    }
                }

                output[kernel] = arr;
            }

            return output;
        }

        private double[][] MapWeights(double[,] weights)
        {
            var output = new double[weights.GetLength(1)][];

            for (int i = 0; i < weights.GetLength(1); i++)
            {
                output[i] = new double[weights.GetLength(0)];

                for (int j = 0; j < weights.GetLength(0); j++)
                {
                    output[i][j] = weights[j, i];
                }
            }

            return output;
        }
    }
}
