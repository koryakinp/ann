using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Ann.Fingers
{
    public static class ImageWriter
    {
        public static void WriteImages(string folderpath, string destination)
        {
            var writer1 = new BinaryWriter(File.OpenWrite(destination));

            DirectoryInfo d = new DirectoryInfo(folderpath);

            writer1.Write(d.GetFiles().Length);
            writer1.Write(128);

            foreach (var file in d.GetFiles("*.png").OrderByDescending(q => q.Name))
            {
                Image<Rgba32> image = SixLabors.ImageSharp.Image.Load(file.OpenRead());
                for (int i = 0; i < 128; i++)
                {
                    for (int j = 0; j < 128; j++)
                    {
                        writer1.Write(image[i, j].B);
                    }
                }

                writer1.Write(byte.Parse(file.Name[37].ToString()));

                Console.WriteLine(file.Name);
            }

            writer1.Close();
        }
    }
}
