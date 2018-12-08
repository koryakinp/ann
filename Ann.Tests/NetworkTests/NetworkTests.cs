using Ann.Activators;
using Ann.Core.Tests.Utils;
using Ann.Layers.Convolution;
using Ann.Layers.Dense;
using Ann.LossFunctions;
using Ann.Utils;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using System.IO;
using System.Linq;

namespace Ann.Core.Tests.NetworkTests
{
    [TestClass]
    public class NetworkTests
    {
        public Network network;
        private readonly DoubleComparer _comparer;

        public NetworkTests()
        {
            _comparer = new DoubleComparer(6);
        }

        [TestInitialize]
        public void Initialize()
        {
            network = new Network(LossFunctionType.CrossEntropy, new Flat(0.1), 3);
        }

        [TestMethod]
        public void CNNTest()
        {
            network.AddInputLayer(7, 1);
            network.AddConvolutionLayer(2, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddConvolutionLayer(3, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddFlattenLayer();
            network.AddDenseLayer(5, true);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(3, false);
            network.AddSoftMaxLayer();

            TestNetwork(network, "test1");
        }

        [TestMethod]
        public void LoadFromFile()
        {
            network.AddInputLayer(7, 1);
            network.AddConvolutionLayer(2, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddConvolutionLayer(3, 2);
            network.AddPoolingLayer(2);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddFlattenLayer();
            network.AddDenseLayer(5, true);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(3, false);
            network.AddSoftMaxLayer();

            network.RandomizeWeights(0.1);

            network.TrainModel(NetworkTestsData.Input, NetworkTestsData.Labels[0]);
            network.BuildModel().Save("network.json");
            var model = new Model("network.json");
            model.Predict(NetworkTestsData.Input);
        }

        [TestMethod]
        public void MNISTTest()
        {
            network.AddInputLayer(28, 1);
            network.AddConvolutionLayer(16, 5);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddConvolutionLayer(32, 5);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddPoolingLayer(2);
            network.AddFlattenLayer();
            network.AddDenseLayer(1024, true);
            network.AddActivationLayer(ActivatorType.Relu);
            network.AddDenseLayer(10, false);
            network.AddSoftMaxLayer();

            TestNetwork(network, "test2");
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

        private double[,,] ReadInput(string file)
        {
            var json1 = File.ReadAllText($"NetworkTests/{file}");
            var w1 = JsonConvert.DeserializeObject<double[,,,]>(json1);

            var output = new double[1, w1.GetLength(1), w1.GetLength(2)];

            for (int j = 0; j < w1.GetLength(1); j++)
            {
                for (int k = 0; k < w1.GetLength(2); k++)
                {
                    output[0, j, k] = w1[0, j, k, 0];
                }
            }

            return output;
        }

        private bool[] ReadLabels(string folder)
        {
            var raw = ReadDenseWeightsJson<double[,]>($"{folder}/data/y.json");
            var output = new bool[raw.Length];
            output.UpdateForEach<bool>((q,i) => (double)raw.GetValue(0,i[0]) == 1);
            return output;
        }

        private void TestNetwork(Network network, string folder)
        {
            network.SetWeights(0, ReadConvWeightsJson($"{folder}/before/conv_1_weights.json"));
            network.SetBiases(0, ReadDenseWeightsJson<double[]>($"{folder}/before/conv_1_biases.json"));
            network.SetWeights(1, ReadConvWeightsJson($"{folder}/before/conv_2_weights.json"));
            network.SetBiases(1, ReadDenseWeightsJson<double[]>($"{folder}/before/conv_2_biases.json"));
            network.SetWeights(2, ReadDenseWeightsJson<double[,]>($"{folder}/before/dense_1_weights.json"));
            network.SetBiases(2, ReadDenseWeightsJson<double[]>($"{folder}/before/dense_1_biases.json"));
            network.SetWeights(3, ReadDenseWeightsJson<double[,]>($"{folder}/before/dense_2_weights.json"));

            var x = ReadInput($"{folder}/data/x.json");
            var y = ReadLabels(folder);
            network.TrainModel(x, y);

            var learnableLayers = network._layers.OfType<ILearnable>().ToArray();
            var conv1layer = learnableLayers[0] as ConvolutionFullLayer;
            var conv2layer = learnableLayers[1] as ConvolutionFullLayer;
            var dense1layer = learnableLayers[2] as DenseFullLayer;
            var dense2layer = learnableLayers[3] as DenseFullLayer;

            var w1 = conv1layer.GetWeights();
            var w2 = conv2layer.GetWeights();
            var w3 = dense1layer.GetWeights();
            var w4 = dense2layer.GetWeights();

            var b1 = conv1layer.GetBiases();
            var b2 = conv2layer.GetBiases();
            var b3 = dense1layer.GetBiases();

            var b1u = ReadDenseWeightsJson<double[]>($"{folder}/after/conv_1_biases.json");
            var b2u = ReadDenseWeightsJson<double[]>($"{folder}/after/conv_2_biases.json");
            var b3u = ReadDenseWeightsJson<double[]>($"{folder}/after/dense_1_biases.json");
            var w1u = ReadConvWeightsJson($"{folder}/after/conv_1_weights.json");
            var w2u = ReadConvWeightsJson($"{folder}/after/conv_2_weights.json");
            var wd1u = ReadDenseWeightsJson<double[,]>($"{folder}/after/dense_1_weights.json");
            var wd2u = ReadDenseWeightsJson<double[,]>($"{folder}/after/dense_2_weights.json");

            CollectionAssert.AreEqual(b1, b1u, _comparer);
            CollectionAssert.AreEqual(b2, b2u, _comparer);
            CollectionAssert.AreEqual(b3, b3u, _comparer);

            w1.ForEach((q, i) => CollectionAssert.AreEqual(q, w1u[i], _comparer));
            w2.ForEach((q, i) => CollectionAssert.AreEqual(q, w2u[i], _comparer));

            CollectionAssert.AreEqual(w3, wd1u, _comparer);
            CollectionAssert.AreEqual(w4, wd2u, _comparer);
        }
    }
}
