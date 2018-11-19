using Ann.Misc;
using Ann.Utils;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ann
{
    public static class MatrixHelper
    {
        public static double[][,,] Transpose(this double[][,,] kernels)
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

        public static double[,,] Convolution(double[][,,] kernels, double[,,] volume)
        {
            foreach (var item in kernels)
            {
                ValidateConvolution(volume, item);
            }

            var kernel = kernels[0];

            int numberOfKernels = kernels.Length;
            int kernelSize = kernel.GetLength(1);
            int volumeSize = volume.GetLength(1);
            int kernelArea = kernelSize * kernelSize;
            int kernelLength = kernel.GetLength(0) * kernel.GetLength(1) * kernel.GetLength(2);
            int outputSize = volumeSize - kernelSize + 1;
            int kernelsPerChannel = outputSize * outputSize;
            int depth = kernel.GetLength(0);

            double[,] temp1 = new double[numberOfKernels, kernelLength];
            double[,] temp2 = new double[kernelArea * depth, outputSize * outputSize];

            var kernelMatrix = Matrix.Build.DenseOfArray(temp1);
            var volumeMatrix = Matrix.Build.DenseOfArray(temp2);

            int row = 0;
            int col = 0;

            for (int z = 0; z < volume.GetLength(0); z++)
            {
                for (int y = 0; y <= volume.GetLength(1) - kernelSize; y++)
                {
                    for (int x = 0; x <= volume.GetLength(2) - kernelSize; x++)
                    {
                        
                        for (int ky = 0; ky < kernelSize; ky++)
                        {
                            for (int kx = 0; kx < kernelSize; kx++)
                            {
                                volumeMatrix[z * kernelArea + row, col] = volume[z, y + ky, x + kx];
                                row++;
                            }
                        }
                        row = 0;
                        col++;
                    }
                }
                col = 0;
            }

            for (int i = 0; i < kernels.Length; i++)
            {
                kernelMatrix.SetRow(i, kernels[i].Cast<double>().ToArray());
            }

            var res = kernelMatrix.Multiply(volumeMatrix);

            var output = new double[res.RowCount, outputSize, outputSize];

            for (int i = 0; i < res.RowCount; i++)
            {
                var arr = res.Row(i).ToArray();
                var temp = Matrix.Build.Dense(outputSize, outputSize, arr).Transpose().ToArray();
                output.SetChannel(temp, i);
            }

            return output;
        }

        public static double[,] Convolution(this double[,] volume, double[,] kernel)
        {
            int size = kernel.GetLength(0);
            int volumeSize = volume.GetLength(1);
            var output = new double[volumeSize - size + 1, volumeSize - size + 1];
            double[] kernelVector = new double[kernel.Length];
            double[] volumeVector = new double[kernel.Length];

            kernel.ForEach((q, j, i) => kernelVector[j * size + i] = q);

            var vector1 = new DenseVector(kernelVector);

            for (int x = 0; x < output.GetLength(0); x++)
            {
                for (int y = 0; y < output.GetLength(1); y++)
                {
                    var temp = new List<double>();
                    for (int j = x; j < x + size; j++)
                    {
                        for (int i = y; i < y + size; i++)
                        {
                            temp.Add(volume[j, i]);
                        }
                    }
                    volumeVector = temp.ToArray();
                    var vector2 = new DenseVector(volumeVector);

                    output[x, y] = vector1.DotProduct(vector2);
                }
            }

            return output;
        }

        public static double[,,] Rotate(this double[,,] input)
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

        public static double[,,] Pad(this double[,,] input, int size)
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
