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
            Assert.Equal(Math.Round(a1.CalculateValue(-2),3), 0.119);
            Assert.Equal(Math.Round(a1.CalculateValue(-1), 3), 0.269);
            Assert.Equal(Math.Round(a1.CalculateValue(0), 3), 0.500);
            Assert.Equal(Math.Round(a1.CalculateValue(1), 3), 0.731);
            Assert.Equal(Math.Round(a1.CalculateValue(2), 3), 0.881);
        }

        [Fact]
        public void LogisticActivatorDeriviativeTest()
        {
            LogisticActivator a1 = new LogisticActivator();
            Assert.Equal(a1.CalculateDeriviative(-2), -6);
            Assert.Equal(a1.CalculateDeriviative(-1), -2);
            Assert.Equal(a1.CalculateDeriviative(0), 0);
            Assert.Equal(a1.CalculateDeriviative(1), 0);
            Assert.Equal(a1.CalculateDeriviative(2), -2);
        }

        [Fact]
        public void TanhActivatorValueTest()
        {
            TanhActivator a1 = new TanhActivator();
            Assert.Equal(Math.Round(a1.CalculateValue(-2), 3), -0.964);
            Assert.Equal(Math.Round(a1.CalculateValue(-1), 3), -0.762);
            Assert.Equal(Math.Round(a1.CalculateValue(0), 3), 0);
            Assert.Equal(Math.Round(a1.CalculateValue(1), 3), 0.762);
            Assert.Equal(Math.Round(a1.CalculateValue(2), 3), 0.964);
        }

        [Fact]
        public void TanhActivatorDeriviativeTest()
        {
            TanhActivator a1 = new TanhActivator();
            Assert.Equal(Math.Round(a1.CalculateDeriviative(-2), 3), 0.071);
            Assert.Equal(Math.Round(a1.CalculateDeriviative(-1), 3), 0.420);
            Assert.Equal(Math.Round(a1.CalculateDeriviative(0), 3), 1);
            Assert.Equal(Math.Round(a1.CalculateDeriviative(1), 3), 0.420);
            Assert.Equal(Math.Round(a1.CalculateDeriviative(2), 3), 0.071);
        }

        [Fact]
        public void ReluActivatorValueTest()
        {
            ReluActivator a1 = new ReluActivator();
            Assert.Equal(a1.CalculateValue(-2), 0);
            Assert.Equal(a1.CalculateValue(-1), 0);
            Assert.Equal(a1.CalculateValue(0), 0);
            Assert.Equal(a1.CalculateValue(1), 1);
            Assert.Equal(a1.CalculateValue(2), 2);
        }

        [Fact]
        public void ReluActivatorDeriviativeTest()
        {
            ReluActivator a1 = new ReluActivator();
            Assert.Equal(a1.CalculateDeriviative(-2), 0);
            Assert.Equal(a1.CalculateDeriviative(-1), 0);
            Assert.Equal(a1.CalculateDeriviative(0), 1);
            Assert.Equal(a1.CalculateDeriviative(1), 1);
            Assert.Equal(a1.CalculateDeriviative(2), 1);
        }
    }
}
