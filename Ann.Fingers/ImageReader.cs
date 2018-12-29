using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace Ann.Fingers
{
    public static class ImageReader
    {
        public static IEnumerable<Image> ReadTrainingData(int max)
        {
            var stream = GetStream("TrainImages");

            var reader = new BinaryReader(stream);

            foreach (var item in Read(reader, max))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData(int max)
        {
            var stream = GetStream("TestImages");

            var reader = new BinaryReader(stream);
            foreach (var item in Read(reader, max))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadMovieData(int max)
        {
            var stream = GetStream("ValidateImages");

            var reader = new BinaryReader(stream);
            foreach (var item in Read(reader, max))
            {
                yield return item;
            }
        }

        private static Stream GetStream(string entry)
        {
            return ZipFile
                .OpenRead($"Images/{entry}.zip")
                .GetEntry(entry)
                .Open();
        }

        private static IEnumerable<Image> Read(BinaryReader reader, int max)
        {
            int imageCount = reader.ReadInt32();
            int imageSize = reader.ReadInt32();

            if (max > imageCount)
            {
                max = imageCount;
            }

            for (int i = 0; i < imageCount && i <= max; i++)
            {
                var data = reader.ReadBytes(imageSize * imageSize);
                var label = reader.ReadByte();
                yield return new Image(data, label);
            }
        }
    }
}
