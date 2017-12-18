using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;

namespace Ann.Utils
{
    public static class ImageProvider
    {
        public static List<Image> ProvideImages(string filepath)
        {
            List<Image> testImages = new List<Image>();

            using (ZipArchive za = ZipFile.OpenRead(filepath))
            {
                Stream stream = za.Entries.First().Open();

                using (var reader = new StreamReader(stream))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        testImages.Add(new Image
                        {
                            Value = byte.Parse(line.Split(',').First()),
                            Data = line.Split(',').Skip(1).Select(q => byte.Parse(q)).ToList()
                        });
                    }
                }
            }

            return testImages;
        }
    }
}
