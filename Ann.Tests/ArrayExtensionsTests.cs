using Ann.Core.Tests.Utils;
using Ann.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Ann.Tests
{
    [TestClass]
    public class ArrayExtensionsTests
    {
        public double[,,] Values { get; set; }

        public ArrayExtensionsTests()
        {
            Values = TestValues.Kernels;
        }

        [TestMethod]
        public void UpdateForEachTest()
        {
            var expected = new double[Values.GetLength(0), Values.GetLength(1), Values.GetLength(2)];

            for (int i = 0; i < Values.GetLength(0); i++)
            {
                for (int j = 0; j < Values.GetLength(1); j++)
                {
                    for (int k = 0; k < Values.GetLength(2); k++)
                    {
                        expected[i, j, k] = Math.Pow(Values[i, j, k], 2);
                    }
                }
            }

            Values.UpdateForEach<double>(q => Math.Pow(q, 2));
            CollectionAssert.AreEqual(expected, Values);
        }

        [TestMethod]
        public void UpdateForEachWidthIndexTest()
        {
            var expected = new double[Values.GetLength(0), Values.GetLength(1), Values.GetLength(2)];

            for (int i = 0; i < Values.GetLength(0); i++)
            {
                for (int j = 0; j < Values.GetLength(1); j++)
                {
                    for (int k = 0; k < Values.GetLength(2); k++)
                    {
                        expected[i, j, k] = Math.Pow(Values[i, j, k], 2);
                    }
                }
            }

            Values.UpdateForEach<double>((q,idx) =>
            {
                Assert.AreEqual(q, Values[idx[0],idx[1],idx[2]]);
                return Math.Pow(q, 2);
            });

            CollectionAssert.AreEqual(expected, Values);
        }
    }
}
