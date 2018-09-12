using Ann.Core.Tests.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ann.Core.Tests
{

    [TestClass]
    public class ArrayConverterTests
    {
        private readonly double[] Convert1Dto4DTestInput;
        private readonly double[] Convert1Dto3DTestInput;

        private readonly double[,,,] Convert1Dto4DTestOutput;
        private readonly double[,,] Convert1Dto3DTestOutput;

        private readonly double[][,,] Convert1Dto3DJaggedTestOutput;

        public ArrayConverterTests()
        {
            Convert1Dto4DTestInput = new double[]
            {
                1111, 1112, 1113,
                1121, 1122, 1123,
                1131, 1132, 1133,

                1211, 1212, 1213,
                1221, 1222, 1223,
                1231, 1232, 1233,

                1311, 1312, 1313,
                1321, 1322, 1323,
                1331, 1332, 1333,


                2111, 2112, 2113,
                2121, 2122, 2123,
                2131, 2132, 2133,

                2211, 2212, 2213,
                2221, 2222, 2223,
                2231, 2232, 2233,

                2311, 2312, 2313,
                2321, 2322, 2323,
                2331, 2332, 2333
            };

            Convert1Dto3DTestInput = new double[]
            {
                111, 112, 113,
                121, 122, 123,
                131, 132, 133,

                211, 212, 213,
                221, 222, 223,
                231, 232, 233,

                311, 312, 313,
                321, 322, 323,
                331, 332, 333
            };

            Convert1Dto3DTestOutput = new double[,,]
            {
                {
                    { 111, 112, 113 },
                    { 121, 122, 123 },
                    { 131, 132, 133 }
                },
                {
                    { 211, 212, 213 },
                    { 221, 222, 223 },
                    { 231, 232, 233 }
                },
                {
                    { 311, 312, 313 },
                    { 321, 322, 323 },
                    { 331, 332, 333 }
                }
            };

            Convert1Dto4DTestOutput = new double[2, 3, 3, 3]
            {
                {
                    {
                        { 1111, 1112, 1113 },
                        { 1121, 1122, 1123 },
                        { 1131, 1132, 1133 }
                    },
                    {

                        { 1211, 1212, 1213 },
                        { 1221, 1222, 1223 },
                        { 1231, 1232, 1233 }
                    },
                    {
                        { 1311, 1312, 1313 },
                        { 1321, 1322, 1323 },
                        { 1331, 1332, 1333 }
                    }
                },
                {
                    {
                        { 2111, 2112, 2113 },
                        { 2121, 2122, 2123 },
                        { 2131, 2132, 2133 }
                    },
                    {
                        { 2211, 2212, 2213 },
                        { 2221, 2222, 2223 },
                        { 2231, 2232, 2233 }
                    },
                    {
                        { 2311, 2312, 2313 },
                        { 2321, 2322, 2323 },
                        { 2331, 2332, 2333 }
                    }
                }
            };

            Convert1Dto3DJaggedTestOutput = new double[2][,,]
            {
                new double[3,3,3]
                {
                    {
                        { 1111, 1112, 1113 },
                        { 1121, 1122, 1123 },
                        { 1131, 1132, 1133 }
                    },
                    {

                        { 1211, 1212, 1213 },
                        { 1221, 1222, 1223 },
                        { 1231, 1232, 1233 }
                    },
                    {
                        { 1311, 1312, 1313 },
                        { 1321, 1322, 1323 },
                        { 1331, 1332, 1333 }
                    }
                },
                new double[3,3,3]
                {
                    {
                        { 2111, 2112, 2113 },
                        { 2121, 2122, 2123 },
                        { 2131, 2132, 2133 }
                    },
                    {
                        { 2211, 2212, 2213 },
                        { 2221, 2222, 2223 },
                        { 2231, 2232, 2233 }
                    },
                    {
                        { 2311, 2312, 2313 },
                        { 2321, 2322, 2323 },
                        { 2331, 2332, 2333 }
                    }
                }
            };
        }

        [TestMethod]
        public void Convert1Dto3DTest()
        {
            var actual = ArrayConverter.Convert1Dto3D(Convert1Dto3DTestInput, new int[] { 3, 3, 3 });
            CollectionAssert.AreEqual(Convert1Dto3DTestOutput, actual);
        }

        [TestMethod]
        public void Convert1Dto4DTest()
        {
            var actual = ArrayConverter.Convert1Dto4D(Convert1Dto4DTestInput, new int[] { 2, 3, 3, 3 });
            CollectionAssert.AreEqual(Convert1Dto4DTestOutput, actual);
        }

        [TestMethod]
        public void Convert1Dto3DjaggedTest()
        {
            var actual = ArrayConverter.ConvertToJagged3D(Convert1Dto4DTestInput, 2, new int[] { 3, 3, 3 });

            Assert.AreEqual(actual.Length, Convert1Dto3DJaggedTestOutput.Length);

            for (int i = 0; i < actual.Length; i++)
            {
                CollectionAssert.AreEqual(Convert1Dto3DJaggedTestOutput[i], actual[i]);
            }
        }
    }
}
