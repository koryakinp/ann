using Ann.Activators;
using Ann.Core.LossFunctions;
using Gdo.Optimizers;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualStudio.TestTools.UnitTesting;
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

        [TestInitialize]
        public void Initialize()
        {
            network = new Network(LossFunctionType.CrossEntropy, 3);
        }

        [TestMethod]
        public void PerceptronTest()
        {
            network.AddInputLayer(9);
            network.AddHiddenLayer(10, ActivatorType.Sigmoid, new Flat(0.5));
            network.AddSoftMaxLayer(new Flat(0.5));

            var arr1 = Vector.Build.Random(9).ToArray();
            var arr2 = Vector.Build.Random(9).ToArray();
            var arr3 = Vector.Build.Random(9).ToArray();

            var goal1 = new bool[3] { true, false, false };
            var goal2 = new bool[3] { false, true, false };
            var goal3 = new bool[3] { false, false, true };

            network.RandomizeWeights();

            for (int i = 0; i < 1000; i++)
            {
                network.TrainModel(arr1, goal1);
                network.TrainModel(arr2, goal2);
                network.TrainModel(arr3, goal3);
            }

            var res1 = network.UseModel(arr1);
            Assert.AreEqual(0, Array.IndexOf(res1, res1.Max()));
            var res2 = network.UseModel(arr2);
            Assert.AreEqual(1, Array.IndexOf(res2, res2.Max()));
            var res3 = network.UseModel(arr3);
            Assert.AreEqual(2, Array.IndexOf(res3, res3.Max()));
        }
    }
}
