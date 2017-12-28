using Ann.Activators;
using Ann.Configuration;
using Ann.Connections;
using Ann.Neurons;
using Ann.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Ann.Tests
{
    public class LearningRateTests
    {
        [Fact]
        public void ShouldWeightBeModifiedWithMomentum()
        {
            Coordinate c1 = new Coordinate(0, 0);
            Coordinate c2 = new Coordinate(0, 1);
            Neuron n = new Neuron(1, new LogisticActivator(), new DefaultWeightInitializer());
            NetworkConfiguration nc = new NetworkConfiguration(new LayerConfiguration())
            {
                LearningRate = 0.1,
                Momentum = 0.9
            };

            NeuronConnection c = new NeuronConnection(new Connection(c1, c2), n);
            c.SetWeight(1);

            c.UpdateWeight(2, nc);
            Assert.Equal(0.8, c.GetWeight(), 10);

            c.UpdateWeight(3, nc);
            Assert.Equal(0.32, c.GetWeight(), 10);

            c.UpdateWeight(4, nc);
            Assert.Equal(-0.512, c.GetWeight(), 10);

            c.UpdateWeight(5, nc);
            Assert.Equal(-1.7608, c.GetWeight(), 10);

        }
    }
}
