using Ann.Core.Misc;
using Ann.Utils;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann.Core
{
    public static class MatrixHelper
    {
        public static double[][,,] Transpose(double[][,,] kernels)
        {
            if(!kernels.Any())
            {
                throw new Exception(Consts.MatrixHelperMessages.TransposeNoKernelsFound);
            }
            else if(kernels.Any(q => q.GetLength(0) != kernels[0].GetLength(0)))
            {
                throw new Exception(Consts.MatrixHelperMessages.TransposeDimensionsInvalid);
            }

            double[][,,] output = new double[kernels[0].GetLength(0)][,,];

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = new double[kernels.Length, kernels[0].GetLength(1), kernels[0].GetLength(2)];
                for (int d = 0; d < output[i].GetLength(0); d++)
                {
                    for (int w = 0; w < output[i].GetLength(1); w++)
                    {
                        for (int h = 0; h < output[i].GetLength(2); h++)
                        {
                            output[i][d, w, h] = kernels[d][i, w, h];
                        }
                    }
                }
            }

            return output;
        }

        private static void ValidateConvolution(double[,,] volume, double[,,] kernel)
        {
            if (kernel.GetLength(1) != kernel.GetLength(2))
            {
                throw new Exception(Consts.MatrixHelperMessages.ConvolutionKernelDimensionsInvalid);
            }
            else if (volume.GetLength(1) != volume.GetLength(2))
            {
                throw new Exception(Consts.MatrixHelperMessages.ConvolutionVolumeDimensionsInvalid);
            }
            else if (volume.GetLength(0) != kernel.GetLength(0))
            {
                throw new Exception(Consts.MatrixHelperMessages.ConvolutionDeptheInvalid);
            }
            else if (volume.GetLength(1) < kernel.GetLength(1))
            {
                throw new Exception(Consts.MatrixHelperMessages.ConvolutionVolumeSizeTooSmall);
            }
        }

        public static double[,] Convolution(double[,,] volume, double[,,] kernel)
        {
            ValidateConvolution(volume, kernel);
            int size = kernel.GetLength(1);
            var output = new double[size, size];
            double[] kernelVector = new double[kernel.Length];
            double[] volumeVector = new double[kernel.Length];

            int height = kernel.GetLength(0);
            int width = kernel.GetLength(1);

            kernel.ForEach((q,k,j,i) => kernelVector[i + width * (j + height * k)] = q);

            var vector1 = new DenseVector(kernelVector);

            for (int x = 0; x < size; x++)
            {
                for (int y = 0; y < size; y++)
                {
                    var temp = new List<double>();
                    for (int k = 0; k < volume.GetLength(0); k++)
                    {
                        for (int j = x; j < x + size; j++)
                        {
                            for (int i = y; i < y + size; i++)
                            {
                                temp.Add(volume[k, j, i]);
                            }
                        }
                    }
                    volumeVector = temp.ToArray();
                    var vector2 = new DenseVector(volumeVector);

                    output[x,y] = vector1.DotProduct(vector2);
                }
            }

            return output;
        }

        public static double[,,] Rotate(double[,,] input)
        {
            if(input.GetLength(1) != input.GetLength(2))
            {
                throw new Exception(Consts.MatrixHelperMessages.RotateKernelDimensionsInvalid);
            }

            var output = new double[
                input.GetLength(0),
                input.GetLength(1),
                input.GetLength(2)];

            int n = input.GetLength(1);

            for (int d = 0; d < input.GetLength(0); d++)
            {
                for (int w = 0; w < input.GetLength(1); w++)
                {
                    for (int h = 0; h < input.GetLength(2); h++)
                    {
                        output[d, h, n - w - 1] = input[d, n - h - 1, w];
                    }
                }
            }

            return output;
        }

        public static double[,,] Pad(double[,,] input, int size)
        {
            if(size < 0)
            {
                throw new Exception(Consts.MatrixHelperMessages.PadPaddingValueInvalid);
            }

            var output = new double[
                input.GetLength(0),
                input.GetLength(1) + size * 2,
                input.GetLength(2) + size * 2];

            for (int k = 0; k < input.GetLength(0); k++)
            {
                for (int i = 0; i < input.GetLength(1); i++)
                {
                    for (int j = 0; j < input.GetLength(2); j++)
                    {
                        output[k, i + size, j + size] = (double)input.GetValue(k, i, j);
                    }
                }
            }

            return output;
        }

        public static MaxPoolResult MaxPool(double[,,] input, int stride)
        {
            if (stride < 2)
            {
                throw new Exception(Consts.MatrixHelperMessages.MaxPoolingStrideInvalid);
            }
            else if(input.GetLength(1) != input.GetLength(2))
            {
                throw new Exception(Consts.MatrixHelperMessages.MaxPoolingDimensionsInvalid);
            }

            int size = input.GetLength(1) % stride == 0
                ? input.GetLength(1) / stride
                : (input.GetLength(1) / stride) + 1;

            var values = new double[input.GetLength(0), size, size];
            var cache = new bool[input.GetLength(0), input.GetLength(1), input.GetLength(2)];

            values.ForEach((i, j, k) =>
            {
                int curX = 0;
                int curY = 0;
                double max = double.MinValue;

                for (int x = j * stride; x < (j + 1) * stride && x < input.GetLength(1); x++)
                {
                    for (int y = k * stride; y < (k + 1) * stride && y < input.GetLength(2); y++)
                    {
                        if (input[i, x, y] > max)
                        {
                            curX = x;
                            curY = y;
                            max = input[i, x, y];
                        }
                    }
                }

                values[i, j, k] = max;
                cache[i, curX, curY] = true;

            });

            return new MaxPoolResult(values, cache);
        }
    }
}
