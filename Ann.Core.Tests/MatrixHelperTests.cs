using Ann.Utils;
using Gdo;
using Gdo.Optimizers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace Ann.Core.Tests
{
    [TestClass]
    public class MatrixHelperTests
    {
        private readonly double[,,] _featureMaps;
        private readonly Optimizer[,,,] _kernels;

        public MatrixHelperTests()
        {
            _featureMaps = new double[,,]
            {
                {
                    { 1,1,0,2,1 },
                    { 0,2,1,0,1 },
                    { 1,0,1,1,0 },
                    { 1,2,0,2,2 },
                    { 0,0,1,0,2 }
                },
                {
                    { 1,2,0,0,2 },
                    { 0,2,0,2,0 },
                    { 2,1,2,2,0 },
                    { 1,2,1,1,0 },
                    { 0,1,2,0,1 }
                },
                {
                    { 1,2,2,2,1 },
                    { 2,1,0,1,2 },
                    { 0,1,0,1,0 },
                    { 1,0,2,2,0 },
                    { 2,0,0,2,1 }
                }
            };
            var temp = new double[,,,]
            {
                {
                    {
                        { -1, 1, -1 },
                        { 1, 0, 0 },
                        { -1, 1, 0 }
                    },
                    {
                        { 1, -1, 1 },
                        { -1, 1, 0 },
                        { 0, -1, -1 }
                    },
                    {
                        { 0, 1, 0 },
                        { 0, 0, -1 },
                        { 1, 0, 0 }
                    }
                },
                {
                    {
                        { -1, -1, 0 },
                        { -1, -1, 1 },
                        { 0, 0, 0 }
                    },
                    {
                        { -1, -1, 0 },
                        { -1, 1, 0 },
                        { 0, 1, -1 }
                    },
                    {
                        { 1, -1, -1 },
                        { 1, 1, -1 },
                        { -1, -1, 0 }
                    }
                },
                {
                    {
                        { -1, 0, 0 },
                        { 1, -1, -1 },
                        { 1, -1, -1 }
                    },
                    {
                        {0, 0, -1 },
                        {1, 0, -1 },
                        {-1, -1, -1 }
                    },
                    {
                        { 0, 0, -1 },
                        { 1, 1, -1 },
                        { -1, -1, -1 }
                    }
                }
            };
            _kernels = new Optimizer[temp.GetLength(0), temp.GetLength(1), temp.GetLength(2), temp.GetLength(3)];
            _kernels.ForEach((k, d, w, h) =>
            {
                var optimizer = new Flat(TestValues.LR);
                optimizer.SetValue(temp[k, d, w, h]);
                _kernels[k, d, w, h] = optimizer;
            });
        }

        [TestMethod]
        public void TestGetReceptiveFieldVectors()
        {
            var actual1 = MatrixHelper.GetReceptiveFieldVectors(_featureMaps, 3, 0, 0);
            var actual2 = MatrixHelper.GetReceptiveFieldVectors(_featureMaps, 3, 2, 2);
            var actual3 = MatrixHelper.GetReceptiveFieldVectors(_featureMaps, 3, 1, 0);

            var expected1 = new List<double[]>
            {
                new double[] { 1,1,0,0,2,1,1,0,1 },
                new double[] { 1,2,0,0,2,0,2,1,2 },
                new double[] { 1,2,2,2,1,0,0,1,0 }
            };
            var expected2 = new List<double[]>()
            {
                new double[] { 1,1,0,0,2,2,1,0,2 },
                new double[] { 2,2,0,1,1,0,2,0,1 },
                new double[] { 0,1,0,2,2,0,0,2,1 }
            };
            var expected3 = new List<double[]>()
            {
                new double[] { 1,0,2,2,1,0,0,1,1 },
                new double[] { 2,0,0,2,0,2,1,2,2 },
                new double[] { 2,2,2,1,0,1,1,0,1 }
            };      

            Assert.AreEqual(expected1.Count, actual1.Count);
            Assert.AreEqual(expected2.Count, actual2.Count);
            Assert.AreEqual(expected3.Count, actual3.Count);

            expected1.ForEach((q,i) => CollectionAssert.AreEqual(q, actual1[i]));
            expected2.ForEach((q,i) => CollectionAssert.AreEqual(q, actual2[i]));
            expected3.ForEach((q,i) => CollectionAssert.AreEqual(q, actual3[i]));
        }

        [TestMethod]
        public void TestGetKernelVectors()
        {
            var actual1 = MatrixHelper.GetKernelVectors(_kernels.Values(), 0);
            var actual2 = MatrixHelper.GetKernelVectors(_kernels.Values(), 1);
            var actual3 = MatrixHelper.GetKernelVectors(_kernels.Values(), 2);

            var expected1 = new List<double[]>()
            {
                new double[] { -1,1,-1,1,0,0,-1,1,0 },
                new double[] { 1,-1,1,-1,1,0,0,-1,-1 },
                new double[] { 0,1,0,0,0,-1,1,0,0 }
            };

            var expected2 = new List<double[]>()
            {
                new double[] { -1,-1,0,-1,-1,1,0,0,0 },
                new double[] { -1,-1,0,-1,1,0,0,1,-1 },
                new double[] { 1,-1,-1,1,1,-1,-1,-1,0 }
            };

            var expected3 = new List<double[]>()
            {
                new double[] { -1,0,0,1,-1,-1,1,-1,-1 },
                new double[] { 0,0,-1,1,0,-1,-1,-1,-1 },
                new double[] { 0,0,-1,1,1,-1,-1,-1,-1 }
            };

            Assert.AreEqual(expected1.Count, actual1.Count);
            Assert.AreEqual(expected2.Count, actual2.Count);
            Assert.AreEqual(expected3.Count, actual3.Count);
            expected1.ForEach((q, i) => CollectionAssert.AreEqual(q, actual1[i]));
            expected2.ForEach((q, i) => CollectionAssert.AreEqual(q, actual2[i]));
            expected3.ForEach((q, i) => CollectionAssert.AreEqual(q, actual3[i]));
        }

        [TestMethod]
        public void TestComputeConvolution()
        {
            var rf1 = MatrixHelper.GetReceptiveFieldVectors(_featureMaps, _kernels.GetLength(3), 0, 0);
            var kr1 = MatrixHelper.GetKernelVectors(_kernels.Values(), 0);

            var rf2 = MatrixHelper.GetReceptiveFieldVectors(_featureMaps, _kernels.GetLength(3), 1, 0);
            var kr2 = MatrixHelper.GetKernelVectors(_kernels.Values(), 1);

            var rf3 = MatrixHelper.GetReceptiveFieldVectors(_featureMaps, _kernels.GetLength(3), 1, 1);
            var kr3 = MatrixHelper.GetKernelVectors(_kernels.Values(), 2);

            Assert.AreEqual(MatrixHelper.ComputeConvolution(rf1, kr1), -1);
            Assert.AreEqual(MatrixHelper.ComputeConvolution(rf2, kr2), -11);
            Assert.AreEqual(MatrixHelper.ComputeConvolution(rf3, kr3), -16);
        }

        [TestMethod]
        public void TestConvolution()
        {
            var actual = MatrixHelper.Convolution(_featureMaps, _kernels.Values());
            var expected = new double[3, 3, 3]
            {
                {
                    { -1,-2,4 },
                    { -1,-1,1 },
                    { 1,-1,-1 }
                },
                {
                    {-6,-11,-1},
                    {-3,-6,-10},
                    {-11,-3,-6}

                },
                {
                    {-9,-11,-9},
                    {-7,-16,-10},
                    {-11,-8,-7}
                }
            };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void TestFlipKernels()
        {
            var actual = MatrixHelper.FilpKernels(_kernels.Values());
            var expected = new double[,,,]
            {
                {
                    {
                        { 0,1,-1 },
                        { 0,0,1 },
                        { -1,1,-1 }
                    },
                    {
                        { -1,-1,0 },
                        { 0,1,-1 },
                        { 1,-1,1 }
                    },
                    {
                        { 0,0,1 },
                        { -1,0,0 },
                        { 0,1,0 }
                    }
                },
                {
                    {
                        { 0,0,0 },
                        { 1,-1,-1 },
                        { 0,-1,-1 }
                    },
                    {
                        { -1,1,0 },
                        { 0,1,-1 },
                        { 0,-1,-1 }
                    },
                    {
                        { 0,-1,-1 },
                        { -1,1,1 },
                        { -1,-1,1 }
                    }
                },
                {
                    {
                        { -1,-1,1 },
                        { -1,-1,1 },
                        { 0,0,-1 }
                    },
                    {
                        { -1,-1,-1 },
                        { -1,0,1 },
                        { -1,0,0 }
                    },
                    {
                        { -1,-1,-1 },
                        { -1,1,1 },
                        { -1,0,0 }

                    }
                }
            };

            CollectionAssert.AreEqual(actual, expected);
        }

        [TestMethod]
        public void TestPad()
        {
            var actual = MatrixHelper.Pad(_featureMaps, 2);
            var expected = new double[,,]
            {
                {
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,1,1,0,2,1,0,0 },
                    { 0,0,0,2,1,0,1,0,0 },
                    { 0,0,1,0,1,1,0,0,0 },
                    { 0,0,1,2,0,2,2,0,0 },
                    { 0,0,0,0,1,0,2,0,0 },
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,0,0,0,0,0,0,0 },
                },
                {
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,1,2,0,0,2,0,0 },
                    { 0,0,0,2,0,2,0,0,0 },
                    { 0,0,2,1,2,2,0,0,0 },
                    { 0,0,1,2,1,1,0,0,0 },
                    { 0,0,0,1,2,0,1,0,0 },
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,0,0,0,0,0,0,0 }

                },
                {
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,1,2,2,2,1,0,0 },
                    { 0,0,2,1,0,1,2,0,0 },
                    { 0,0,0,1,0,1,0,0,0 },
                    { 0,0,1,0,2,2,0,0,0 },
                    { 0,0,2,0,0,2,1,0,0 },
                    { 0,0,0,0,0,0,0,0,0 },
                    { 0,0,0,0,0,0,0,0,0 }
                }
            };

            CollectionAssert.AreEqual(actual, expected);
        }
    }
}
