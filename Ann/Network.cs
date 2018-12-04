using Ann.Layers;
using Ann.LossFunctions;
using Ann.Persistence;
using Ann.Utils;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Ann
{
    public partial class Network
    {
        internal readonly List<Layer> _layers;
        internal readonly List<Layer> _testLayers;
        private readonly LossFunction _lossFunction;
        internal readonly int _numberOfClasses;

        private readonly JsonSerializerSettings jsonSerializerSettings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            Formatting = Formatting.Indented
        };

        public Network(string path)
        {
            var json = File.ReadAllText(path);
            var nc = JsonConvert.DeserializeObject<NetworkConfiguration>(json, jsonSerializerSettings);
            _layers = new List<Layer>();
            foreach (var layerConfig in nc.Layers)
            {
                var layer = LayerFactory.Produce(layerConfig);

                _layers.Add(layer);

            }
        }

        public Network(LossFunctionType lossFunctionType, int numberOfClasses)
        {
            _numberOfClasses = numberOfClasses;
            _lossFunction = LossFunctionFactory.Produce(lossFunctionType);
            _layers = new List<Layer>();
            _testLayers = new List<Layer>();
        }


        public void TrainModel(Array input, bool[] labels)
        {
            double[] output = PassForward(input).Cast<double>().ToArray();
            double[] error = _lossFunction.ComputeDeriviative(labels, output);

            PassBackward(error);
            Learn();
        }

        public double[] UseModel(Array input)
        {
            return PassForward(input).Cast<double>().ToArray();
        }

        private Array PassForward(Array input0)
        {
            Array input_0 = new double[input0.GetLength(0), input0.GetLength(1), input0.GetLength(2)];
            input_0.UpdateForEach<double>((q, idx) => (double)input0.GetValue(idx));

            var output1 = _layers[0].PassForward(input0);
            var output_1 = _testLayers[0].PassForward(input_0);
            var res1 = CompareArrays(output1, output_1);

            var output2 = _layers[1].PassForward(output1);
            var output_2 = _testLayers[1].PassForward(output_1);
            var res2 = CompareArrays(output2, output_2);

            var output3 = _layers[2].PassForward(output2);
            var output_3 = _testLayers[2].PassForward(output_2);
            var res3 = CompareArrays(output3, output_3);

            var output4 = _layers[3].PassForward(output3);
            var output_4 = _testLayers[3].PassForward(output_3);
            var res4 = CompareArrays(output4, output_4);

            var output5 = _layers[4].PassForward(output4);
            var output_5 = _testLayers[4].PassForward(output_4);
            var res5 = CompareArrays(output5, output_5);

            var output6 = _layers[5].PassForward(output5);
            var output_6 = _testLayers[5].PassForward(output_5);
            var res6 = CompareArrays(output6, output_6);

            var output7 = _layers[6].PassForward(output6);
            var output_7 = _testLayers[6].PassForward(output_6);
            var res7 = CompareArrays(output7, output_7);

            var output8 = _layers[7].PassForward(output7);
            var output_8 = _testLayers[7].PassForward(output_7);
            var res8 = CompareArrays(output8, output_8);

            var output9 = _layers[8].PassForward(output8);
            var output_9_1 = _testLayers[8].PassForward(output_8);
            var output_9 = _testLayers[9].PassForward(output_9_1);
            var res9 = CompareArrays(output9, output_9);

            var output10 = _layers[9].PassForward(output9);
            var output_10_1 = _testLayers[10].PassForward(output_9);
            var output_10 = _testLayers[11].PassForward(output_10_1);
            var res10 = CompareArrays(output10, output_10);

            return output_10;
        }

        private bool CompareArrays(Array arr1, Array arr2)
        {
            if(arr1.Rank == 3)
            {
                if (arr1.GetLength(0) != arr2.GetLength(0)
                    || arr1.GetLength(1) != arr2.GetLength(1)
                    || arr1.GetLength(2) != arr2.GetLength(2))
                {
                    return false;
                }

                for (int i = 0; i < arr1.GetLength(0); i++)
                {
                    for (int j = 0; j < arr1.GetLength(1); j++)
                    {
                        for (int k = 0; k < arr1.GetLength(2); k++)
                        {
                            if (Math.Abs((double)arr1.GetValue(i,j,k) - (double)arr2.GetValue(i,j,k)) > 0.000000000001)
                            {
                                return false;
                            }
                        }
                    }
                }
            }
            else
            {
                if (arr1.GetLength(0) != arr2.GetLength(0))
                {
                    return false;
                }

                for (int i = 0; i < arr1.Length; i++)
                {
                    if(Math.Abs((double)arr1.GetValue(i) - (double)arr2.GetValue(i)) > 0.000000000001)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        private void PassBackward(Array error10)
        {
            Array error_10 = new double[error10.Length];
            error_10.UpdateForEach<double>((q, idx) => (double)error10.GetValue(idx));

            var error9 = _layers[9].PassBackward(error10);
            var error_9_1 = _testLayers[11].PassBackward(error_10);
            var error_9 = _testLayers[10].PassBackward(error_9_1);
            var res9 = CompareArrays(error9, error_9);

            var error8 = _layers[8].PassBackward(error9);
            var error_8_0 = _testLayers[9].PassBackward(error_9);
            var error_8 = _testLayers[8].PassBackward(error_8_0);
            var res8 = CompareArrays(error8, error_8);

            var error7 = _layers[7].PassBackward(error8);
            var error_7 = _testLayers[7].PassBackward(error_8);
            var res7 = CompareArrays(error7, error_7);

            var error6 = _layers[6].PassBackward(error7);
            var error_6 = _testLayers[6].PassBackward(error_7);
            var res6 = CompareArrays(error6, error_6);

            var error5 = _layers[5].PassBackward(error6);
            var error_5 = _testLayers[5].PassBackward(error_6);
            var res5 = CompareArrays(error5, error_5);

            var error4 = _layers[4].PassBackward(error5);
            var error_4 = _testLayers[4].PassBackward(error_5);
            var res4 = CompareArrays(error4, error_4);

            var error3 = _layers[3].PassBackward(error4);
            var error_3 = _testLayers[3].PassBackward(error_4);
            var res3 = CompareArrays(error3, error_3);

            var error2 = _layers[2].PassBackward(error3);
            var error_2 = _testLayers[2].PassBackward(error_3);
            var res2 = CompareArrays(error2, error_2);

            var error1 = _layers[1].PassBackward(error2);
            var error_1 = _testLayers[1].PassBackward(error_2);
            var res1 = CompareArrays(error1, error_1);

            var error0 = _layers[0].PassBackward(error1);
            var error_0 = _testLayers[0].PassBackward(error_1);
            var res0 = CompareArrays(error0, error_0);
        }

        private void Learn()
        {
            var oldd = _layers.OfType<ILearnable>().ToArray();
            var neww = _testLayers.OfType<ILearnable>().ToArray();

            oldd[3].UpdateWeights();
            neww[3].UpdateWeights();

            var w3_old = ((SoftMaxLayer)oldd[3]).GetWeights();
            var w3_new = ((DenseLayer)neww[3]).GetWeights();

            var res3 = CompareArrays2(w3_old, w3_new);

            oldd[2].UpdateWeights();
            neww[2].UpdateWeights();

            var w2_old = ((HiddenLayer)oldd[2]).GetWeights();
            var w2_new = ((DenseLayer)neww[2]).GetWeights();

            var res2 = CompareArrays2(w2_old, w2_new);

            oldd[2].UpdateBiases();
            neww[2].UpdateBiases();

            var b2_old = ((HiddenLayer)oldd[2]).GetBiases();
            var b2_new = ((DenseLayer)neww[2]).GetBiases();

            var res22 = CompareArrays3(b2_old, b2_new);

            oldd[1].UpdateWeights();
            neww[1].UpdateWeights();

            var w1_old = ((ConvolutionLayer)oldd[1]).GetWeights();
            var w1_new = ((ConvolutionLayer)neww[1]).GetWeights();

            var res1 = CompareArrays4(w1_old, w1_new);

            oldd[1].UpdateBiases();
            neww[1].UpdateBiases();

            var b1_old = ((ConvolutionLayer)oldd[1]).GetBiases();
            var b1_new = ((ConvolutionLayer)neww[1]).GetBiases();

            var res11 = CompareArrays3(b1_old, b1_new);

            oldd[0].UpdateWeights();
            neww[0].UpdateWeights();

            var w0_old = ((ConvolutionLayer)oldd[0]).GetWeights();
            var w0_new = ((ConvolutionLayer)neww[0]).GetWeights();

            var res00 = CompareArrays4(w0_old, w0_new);

            oldd[0].UpdateBiases();
            neww[0].UpdateBiases();

            var b0_old = ((ConvolutionLayer)oldd[0]).GetBiases();
            var b0_new = ((ConvolutionLayer)neww[0]).GetBiases();

            var res0 = CompareArrays3(b0_old, b0_new);

            var total = res0 && res1 && res2 && res3 && res00 && res11 & res22;

            if(!total)
            {

            }
        }


        private bool CompareArrays4(double[][,,] arr1, double[][,,] arr2)
        {
            for (int kernel = 0; kernel < arr1.Length; kernel++)
            {
                for (int i = 0; i < arr1[kernel].GetLength(0); i++)
                {
                    for (int j = 0; j < arr1[kernel].GetLength(1); j++)
                    {
                        for (int k = 0; k < arr1[kernel].GetLength(2); k++)
                        {
                            var val1 = arr1[kernel][i, j, k];
                            var val2 = arr2[kernel][i, j, k];

                            if (Math.Abs(val1 - val2) > 0.000000000001)
                            {
                                return false;
                            }
                        }
                    }
                }
            }

            return true;
        }

        private bool CompareArrays3(double[] arr1, double[] arr2)
        {
            for (int i = 0; i<arr1.Length; i++)
            {
                var val1 = arr1[i];
                var val2 = arr2[i];

                if (Math.Abs(val1 - val2) > 0.000000000001)
                {
                    return false;
                }
            }
            return true;
        }

        private bool CompareArrays3(double[][] arr1, double[][] arr2)
        {
            for (int i = 0; i < arr1.Length; i++)
            {
                for (int j = 0; j < arr1[i].Length; j++)
                {
                    var val1 = arr1[i][j];
                    var val2 = arr2[i][j];

                    if (Math.Abs(val1 - val2) > 0.000000000001)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        private bool CompareArrays2(double[][] arr1, double[,] arr2)
        {
            for (int i = 0; i < arr2.GetLength(0); i++)
            {
                for (int j = 0; j < arr2.GetLength(1); j++)
                {
                    var val1 = arr1[j][i];
                    var val2 = arr2[i, j];

                    if(Math.Abs(val1 - val2) > 0.000000000001)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        public void RandomizeWeights(double stddev)
        {
            foreach (var layer in _layers.OfType<ILearnable>())
            {
                layer.RandomizeWeights(stddev);
            }
        }

        internal void SetWeights(int layerIndex, Array weights, bool test = true)
        {
            if(test)
            {
                _layers.OfType<ILearnable>().ToArray()[layerIndex].SetWeights(weights);
            }
            else
            {
                _testLayers.OfType<ILearnable>().ToArray()[layerIndex].SetWeights(weights);
            }
        }

        internal void SetBiases(int layerIndex, Array weights, bool test = true)
        {
            if(test)
            {
                _layers.OfType<ILearnable>().ToArray()[layerIndex].SetBiases(weights);
            }
            else
            {
                _testLayers.OfType<ILearnable>().ToArray()[layerIndex].SetBiases(weights);
            }
        }

        public void SaveModel(string path)
        {
            var networkConfig = new NetworkConfiguration();
            foreach (var layer in _layers)
            {
                networkConfig.Layers.Add(layer.GetLayerConfiguration());
            }

            var json = JsonConvert.SerializeObject(networkConfig, jsonSerializerSettings);
            File.WriteAllText(path, json);
        }
    }
}
