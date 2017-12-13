using ImageSharp;
using ImageSharp.Colors.Spaces;
using System;
using System.Collections.Generic;
using System.IO;
using ImageIng = ImageSharp.Image;

namespace Ann.Client
{
    public static class ImageParser
    {
        public static void Parse(Image image)
        {
            using (FileStream fileStream = File.OpenWrite(Path.Combine("images", $"{image.Value}_{Guid.NewGuid()}.jpg")))
            {
                ImageIng img = new ImageIng(28, 28);

                for (int i = 0; i < image.Data.Count; i++)
                {
                    var val = image.Data[i];

                    img.Pixels[i].R = val;
                    img.Pixels[i].G = val;
                    img.Pixels[i].B = val;
                }

                img.SaveAsJpeg(fileStream);
            }
        }

        public static List<byte> ConvertImageToBytes(string file)
        {
            List<byte> bytes = new List<byte>();
            using (FileStream fileStream = File.OpenRead(Path.Combine(file)))
            {
                ImageIng img = new ImageIng(fileStream);

                for (int i = 0; i < img.Width; i++)
                {
                    for (int j = 0;  j < img.Height;  j++)
                    {
                        bytes.Add(img.Pixels[i * img.Width + j].R);
                    }
                }
            }

            return bytes;
        }
    }
}
