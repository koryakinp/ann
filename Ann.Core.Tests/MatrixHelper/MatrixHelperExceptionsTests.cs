using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using static Ann.Core.MatrixHelper;

namespace Ann.Core.Tests.MatrixHelper
{
    [TestClass]
    public class MatrixHelperExceptionsTests
    {
        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.TransposeDimensionsInvalid)]
        public void TransposeShouldThrowIfKernelSizesAreInvalid()
        {
            Transpose(new double[2][,,] 
            {
                new double[1,2,3],
                new double[2,2,3]
            });
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.TransposeNoKernelsFound)]
        public void TransposeShouldThrowIfNoKernelsFound()
        {
            Transpose(new double[0][,,] {  });
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionDeptheInvalid)]
        public void ConvolutionShouldThrowIfKernelDepthAndVolumeDepthDoNotMatch()
        {
            Convolution(new double[3,5,5], new double[10,5,5]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionVolumeSizeTooSmall)]
        public void ConvolutionShouldThrowIfVolumeSizeTooSmall()
        {
            Convolution(new double[3, 5, 5], new double[3, 10, 10]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionVolumeDimensionsInvalid)]
        public void ConvolutionShouldThrowIfVolumeDimesnionsInvalid()
        {
            Convolution(new double[3, 7, 5], new double[3, 3, 3]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionKernelDimensionsInvalid)]
        public void ConvolutionShouldThrowIfKernelDimesnionsInvalid()
        {
            Convolution(new double[3, 7, 7], new double[3, 3, 4]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.RotateKernelDimensionsInvalid)]
        public void RotateShouldThrowIfKernelDimesnionsInvalid()
        {
            Rotate(new double[3, 6, 7]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.RotateKernelDimensionsInvalid)]
        public void PadShouldThrowIfPaddingInvalid()
        {
            Pad(new double[0,0,0], -2);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.MaxPoolingStrideInvalid)]
        public void MaxPoolShouldThrowIfStrideInvalid()
        {
            MaxPool(new double[3, 3, 3], 1);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.MaxPoolingDimensionsInvalid)]
        public void MaxPoolShouldThrowIfDimensionsInvalid()
        {
            MaxPool(new double[3, 4, 3], 5);
        }
    }
}
