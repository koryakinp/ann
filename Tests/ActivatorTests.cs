using Ann.Activators;
using Ann.Neurons;
using System;
using Xunit;

namespace Ann.Tests
{
    public class ActivatorTests
    {
        [Fact]
        public void LogisticActivatorValueTest()
        {
            LogisticActivator a1 = new LogisticActivator();
            Assert.Equal(0.119, a1.CalculateValue(-2),3);
            Assert.Equal(0.269, a1.CalculateValue(-1), 3);
            Assert.Equal(0.500, a1.CalculateValue(0), 3);
            Assert.Equal(0.731, a1.CalculateValue(1), 3);
            Assert.Equal(0.881, a1.CalculateValue(2), 3);
        }

        [Fact]
        public void LogisticActivatorDeriviativeTest()
        {
            LogisticActivator a1 = new LogisticActivator();
            Assert.Equal(-6, a1.CalculateDeriviative(-2));
            Assert.Equal(-2, a1.CalculateDeriviative(-1));
            Assert.Equal(0, a1.CalculateDeriviative(0));
            Assert.Equal(0, a1.CalculateDeriviative(1));
            Assert.Equal(-2, a1.CalculateDeriviative(2));
        }

        [Fact]
        public void TanhActivatorValueTest()
        {
            TanhActivator a1 = new TanhActivator();
            Assert.Equal(-0.964, a1.CalculateValue(-2), 3);
            Assert.Equal(-0.762, a1.CalculateValue(-1), 3);
            Assert.Equal(0, a1.CalculateValue(0), 3);
            Assert.Equal(0.762, a1.CalculateValue(1), 3);
            Assert.Equal(0.964, a1.CalculateValue(2), 3);
        }

        [Fact]
        public void TanhActivatorDeriviativeTest()
        {
            TanhActivator a1 = new TanhActivator();
            Assert.Equal(0.071, a1.CalculateDeriviative(-2), 3);
            Assert.Equal(0.420, a1.CalculateDeriviative(-1), 3);
            Assert.Equal(1, a1.CalculateDeriviative(0), 3);
            Assert.Equal(0.420, a1.CalculateDeriviative(1), 3);
            Assert.Equal(0.071, a1.CalculateDeriviative(2), 3);
        }

        [Fact]
        public void ReluActivatorValueTest()
        {
            ReluActivator a1 = new ReluActivator();
            Assert.Equal(0, a1.CalculateValue(-2));
            Assert.Equal(0, a1.CalculateValue(-1));
            Assert.Equal(0, a1.CalculateValue(0));
            Assert.Equal(1, a1.CalculateValue(1));
            Assert.Equal(2, a1.CalculateValue(2));
        }

        [Fact]
        public void ReluActivatorDeriviativeTest()
        {
            ReluActivator a1 = new ReluActivator();
            Assert.Equal(0, a1.CalculateDeriviative(-2));
            Assert.Equal(0, a1.CalculateDeriviative(-1));
            Assert.Equal(1, a1.CalculateDeriviative(0));
            Assert.Equal(1, a1.CalculateDeriviative(1));
            Assert.Equal(1, a1.CalculateDeriviative(2));
        }
    }
}
