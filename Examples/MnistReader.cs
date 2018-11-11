using System.Collections.Generic;
using System.IO;

namespace Ann.Mnist
{
    public static class MnistReader
    {
        private const string TrainImages = "mnist/train-images.idx3-ubyte";
        private const string TrainLabels = "mnist/train-labels.idx1-ubyte";
        private const string TestImages = "mnist/t10k-images.idx3-ubyte";
        private const string TestLabels = "mnist/t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData(int max)
        {
            foreach (var item in Read(TrainImages, TrainLabels, max))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData(int max)
        {
            foreach (var item in Read(TestImages, TestLabels, max))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath, int max)
        {
            List<Image> imgs = new List<Image>();
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicImages = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();

            int imgCount = numberOfImages > max ? max : numberOfImages;

            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < imgCount; i++)
            {
                yield return new Image()
                {
                    Data = images.ReadBytes(width * height),
                    Label = labels.ReadByte()
                };
            }
        }
    }
}
