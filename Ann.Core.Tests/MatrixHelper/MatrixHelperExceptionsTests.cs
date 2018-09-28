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
            new double[2][,,] 
            {
                new double[1,2,3],
                new double[2,2,3]
            }
            .Transpose();
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.TransposeNoKernelsFound)]
        public void TransposeShouldThrowIfNoKernelsFound()
        {
            new double[0][,,] {  }.Transpose();
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionDeptheInvalid)]
        public void ConvolutionShouldThrowIfKernelDepthAndVolumeDepthDoNotMatch()
        {
            new double[10,5,5].Convolution(new double[3, 5, 5]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionVolumeSizeTooSmall)]
        public void ConvolutionShouldThrowIfVolumeSizeTooSmall()
        {
            new double[3, 5, 5].Convolution(new double[3, 10, 10]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionVolumeDimensionsInvalid)]
        public void ConvolutionShouldThrowIfVolumeDimesnionsInvalid()
        {
            new double[3, 3, 3].Convolution(new double[3, 7, 5]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.ConvolutionKernelDimensionsInvalid)]
        public void ConvolutionShouldThrowIfKernelDimesnionsInvalid()
        {
            new double[3, 7, 7].Convolution(new double[3, 3, 4]);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.RotateKernelDimensionsInvalid)]
        public void RotateShouldThrowIfKernelDimesnionsInvalid()
        {
            new double[3, 6, 7].Rotate();
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), Consts.MatrixHelperMessages.RotateKernelDimensionsInvalid)]
        public void PadShouldThrowIfPaddingInvalid()
        {
            new double[0,0,0].Pad(-2);
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
