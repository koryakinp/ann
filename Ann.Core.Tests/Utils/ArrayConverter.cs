using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ann.Core.Tests.Utils
{
    public static class ArrayConverter
    {
        public const string SizesDoNotMatch = "Sizes do not match";

        public static T[][,,] ConvertToJagged3D<T>(T[] input, int jagged, int[] dimensions)
        {
            var temp = Convert1Dto4D(input, new int[] { jagged, dimensions[0], dimensions[1], dimensions[2] });

            var output = new T[jagged][,,];

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = new T[dimensions[0], dimensions[1], dimensions[2]];

                for (int d = 0; d < temp.GetLength(1); d++)
                {
                    for (int w = 0; w < temp.GetLength(2); w++)
                    {
                        for (int h = 0; h < temp.GetLength(3); h++)
                        {
                            output[i][d,w,h] = temp[i, d, w, h];
                        }
                    }
                }
            }

            return output;
        }

        public static T[,,,] Convert1Dto4D<T>(T[] input, int[] dimensions)
        {
            ValidateSizes(input, dimensions);
            var output = new T[dimensions[0], dimensions[1], dimensions[2], dimensions[3]];

            for (int k = 0; k < dimensions[0]; k++)
            {
                for (int d = 0; d < dimensions[1]; d++)
                {
                    for (int j = 0; j < dimensions[2]; j++)
                    {
                        for (int i = 0; i < dimensions[3]; i++)
                        {
                            output[k, d, j, i] = input[
                                k * dimensions[1] * dimensions[2] * dimensions[3] + 
                                d * dimensions[2] * dimensions[3] + 
                                j * dimensions[3] + 
                                i];
                        }
                    }
                }
            }

            return output;
        }

        public static T[,,] Convert1Dto3D<T>(T[] input, int[] dimensions)
        {
            ValidateSizes(input, dimensions);

            var output = new T[dimensions[0], dimensions[1], dimensions[2]];

            for (int k = 0; k < dimensions[0]; k++)
            {
                for (int d = 0; d < dimensions[1]; d++)
                {
                    for (int j = 0; j < dimensions[2]; j++)
                    {
                        output[k, d, j] = input[
                            k * dimensions[1] * dimensions[2] + 
                            d * dimensions[2] + 
                            j];
                    }
                }
            }

            return output;
        }

        public static T[,] Convert1Dto2D<T>(T[] input, int[] dimensions)
        {
            ValidateSizes(input, dimensions);
            var output = new T[dimensions[0], dimensions[1]];

            for (int k = 0; k < dimensions[0]; k++)
            {
                for (int d = 0; d < dimensions[1]; d++)
                {
                    output[k, d] = input[k * dimensions[0] + d];
                }
            }

            return output;
        }

        private static void ValidateSizes<T>(T[] input, int[] dimensions)
        {
            if (input.Length != dimensions.Aggregate((q, w) => q * w))
            {
                throw new Exception(SizesDoNotMatch);
            }
        }
    }
}
