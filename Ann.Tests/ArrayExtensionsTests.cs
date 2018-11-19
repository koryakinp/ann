using Ann.Core.Tests.Utils;
using Ann.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Ann.Tests
{
    [TestClass]
    public class ArrayExtensionsTests
    {
        public double[,,] values { get; set; }

        public ArrayExtensionsTests()
        {
            values = TestValues.Kernels;
        }

        [TestMethod]
        public void UpdateForEachTest()
        {
            var expected = new double[values.GetLength(0), values.GetLength(1), values.GetLength(2)];

            for (int i = 0; i < values.GetLength(0); i++)
            {
                for (int j = 0; j < values.GetLength(1); j++)
                {
                    for (int k = 0; k < values.GetLength(2); k++)
                    {
                        expected[i, j, k] = Math.Pow(values[i, j, k], 2);
                    }
                }
            }

            values.UpdateForEach<double>(q => Math.Pow(q, 2));
            CollectionAssert.AreEqual(expected, values);
        }

        [TestMethod]
        public void UpdateForEachWidthIndexTest()
        {
            var expected = new double[values.GetLength(0), values.GetLength(1), values.GetLength(2)];

            for (int i = 0; i < values.GetLength(0); i++)
            {
                for (int j = 0; j < values.GetLength(1); j++)
                {
                    for (int k = 0; k < values.GetLength(2); k++)
                    {
                        expected[i, j, k] = Math.Pow(values[i, j, k], 2);
                    }
                }
            }

            values.UpdateForEach<double>((q,idx) =>
            {
                Assert.AreEqual(q, values[idx[0],idx[1],idx[2]]);
                return Math.Pow(q, 2);
            });

            CollectionAssert.AreEqual(expected, values);
        }
    }
}
