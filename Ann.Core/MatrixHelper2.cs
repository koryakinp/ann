using Ann.Utils;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;

namespace Ann.Core
{
    public static class MatrixHelper2
    {
        public static List<double[]> GetReceptiveFieldVectors(Array source, int size, int x, int y)
        {
            if(source.Rank != 3)
            {
                throw new Exception();
            }

            var output = new List<double[]>(source.GetLength(0));
            for (int c = 0; c < source.GetLength(0); c++)
            {
                var temp = new double[size * size];
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        temp[i * size + j] = (double)source.GetValue(c, y + i, x + j);
                    }
                }
                output.Add(temp);
            }

            return output;
        }

        public static List<double[]> GetKernelVectors(double[,,,] source, int kernel)
        {
            var output = new List<double[]>(source.GetLength(1));
            var size = source.GetLength(3);
            for (int c = 0; c < source.GetLength(1); c++)
            {
                var temp = new double[size * size];
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        temp[i * size + j] = source[kernel, c, i, j];
                    }
                }
                output.Add(temp);
            }

            return output;
        }

        public static double ComputeConvolution(List<double[]> receptiveFields, List<double[]> kernels)
        {
            if(receptiveFields.Count != kernels.Count)
            {
                throw new Exception(Consts.KernelDepthIsNotValid);
            }

            double output = 0;

            for (int i = 0; i < receptiveFields.Count; i++)
            {
                var vector1 = new DenseVector(receptiveFields[i]);
                var vector2 = new DenseVector(kernels[i]);

                output += vector1.DotProduct(vector2);
            }

            return output;
        }

        public static double[,,] Convolution(Array featureMaps, double[,,,] kernels)
        {
            if(featureMaps.Rank != 3)
            {
                throw new Exception();
            }

            int numberOfKernels = kernels.GetLength(0);
            int kernelSize = kernels.GetLength(3);
            int featureMapSize = featureMaps.GetLength(2);
            int outputSize = featureMapSize - kernelSize + 1;

            if(featureMapSize < kernelSize)
            {
                throw new Exception(Consts.FeatureMapSmallerThanKernel);
            }

            var output = new double[numberOfKernels, outputSize, outputSize];

            for (int k = 0; k < kernels.GetLength(0); k++)
            {
                var kervelVectors = GetKernelVectors(kernels, k);
                for (int i = 0; i < outputSize; i++)
                {
                    for (int j = 0; j < outputSize; j++)
                    {
                        var receptiveFieldVectors = GetReceptiveFieldVectors(featureMaps, kernelSize, j, i);
                        output[k, i, j] = ComputeConvolution(receptiveFieldVectors, kervelVectors);
                    }
                }
            }

            return output;
        }

        public static double[,,,] FilpKernels(double[,,,] kernels)
        {
            if(kernels.GetLength(2) != kernels.GetLength(3))
            {
                throw new Exception(Consts.CanNotFlipKernels);
            }

            var output = (double[,,,])kernels.Clone();

            Array kernel = new double[2];

            int n = kernels.GetLength(2);

            for (int k = 0; k < kernels.GetLength(0); k++)
            {
                for (int d = 0; d < kernels.GetLength(1); d++)
                {
                    for (int w = 0; w < n; ++w)
                    {
                        for (int h = 0; h < n; ++h)
                        {
                            output[k, d, h, n - w - 1] = kernels[k, d, n - h - 1, w];
                        }
                    }
                }
            }

            return output;
        }

        public static double[,,] Pad(double[,,] input, int size)
        {
            var output = new double[
                input.GetLength(0),
                input.GetLength(1) + size * 2,
                input.GetLength(2) + size * 2];

            input.ForEach((q, k, i, j) => output[k, i + size, j + size] = q);

            return output;
        }
    }
}
