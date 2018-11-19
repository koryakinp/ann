namespace Ann
{
    internal class Consts
    {
        public class MatrixHelperMessages
        {
            public const string TransposeDimensionsInvalid = "Can not transpose kernels. Dimensions are not valid.";
            public const string TransposeNoKernelsFound = "Can not transpose kernels. No kernels found.";
            public const string ConvolutionKernelDimensionsInvalid = "Can not compute convolution. Kernel dimensions are not valid.";
            public const string ConvolutionVolumeDimensionsInvalid = "Can not compute convolution. Volume dimensions are not valid.";
            public const string ConvolutionDeptheInvalid = "Can not compute convolution. Kernel depth and Volume depth do not match.";
            public const string ConvolutionVolumeSizeTooSmall = "Can not compute convolution. Volume size is less than kernel size.";
            public const string RotateKernelDimensionsInvalid = "Can not rotate kernel. Kernel dimensions are not valid.";
            public const string PadPaddingValueInvalid = "Can not apply padding. Padding value is not valid.";
            public const string MaxPoolingStrideInvalid = "Can not compute Max-Pooling result. Stride value is invalid.";
            public const string MaxPoolingDimensionsInvalid = "Can not compute Max-Pooling result. Input dimensions do not match.";
        }

        public class CommonLayerMessages
        {
            public const string CanNotSetWeights = "Can not set weights.Input value is invalid";
            public const string MessageDimenionsInvalid = "Message dimensions are not valid";
        }

        public const string MissingInputLayer = "Can not add layer. An input layer is missing";
        public const string CanNotConvertMessage = "Can not convert single dimensional message to multi dimensional message";
        public const string CanNotSetWeights = "Can not set weights. Number of weights do not match.";
        public const string CanNotUpdateWeights = "Can not update weights. Number of weights do not match.";
        public const string CanNotComputeSoftMax = "Can not compute value. Number of inputs and number of neurins do not match.";
        public const string FeatureMapSmallerThanKernel = "Can not compute convolution. Feature Map size is less than kernel";
        public const string KernelDepthIsNotValid = "Can not compute convolution. Kernel depth is not equal to feature map depth";
        public const string ConvolutionalInputIsNotValid = "Can not perform convolution on a single dimension input";
        public const string MessageShapeIsNotValid = "Message shape is not valid. Message with and height do not match";
        public const string CanNotFlipKernels = "Can not flip kernels. Kernel with and height do not match";
    }
}
