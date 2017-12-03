using System;
using System.Collections.Generic;

namespace Ann
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new Network();

            network
                .AddInputLayer(1)
                .AddHiddenLayer(3)
                .AddOutputLayer(2)
                .Save();

            network.Train(new List<double> { 5 }, new List<double> { 1, 1 });

        }
    }
}
