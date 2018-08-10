namespace Ann.Core
{
    internal class Consts
    {
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
