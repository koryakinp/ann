using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;
using System;
using System.IO;
using System.Linq;

namespace Ann.Fingers
{
    public static class Helper
    {
        public static void Resize(this Image<Rgba32> image, int size)
        {
            image.Mutate(q => q.Resize(new ResizeOptions { Size = new Size(size, size) }));
        }

        public static void SaveImage(this Image<Rgba32> image, string path)
        {
            image.SaveAsJpeg(File.OpenWrite(path));
        }

        public static void DrawPredictions(this Image<Rgba32> image, double[] probabilities)
        {
            FontFamily ff;
            SystemFonts.Collection.TryFind("Arial", out ff);
            var font = new Font(ff, 16f, FontStyle.Regular);

            IPen<Rgba32> pen = new Pen<Rgba32>(Rgba32.White, 2);

            for (int i = 0; i < probabilities.Length; i++)
            {
                image.Mutate(q => q.DrawText($"{i}:", font, Rgba32.White, new PointF(20, 20 + 20 * i)));

                RectangleF shape = new RectangleF(50, 20 + 20 * i, (float)probabilities[i] * 100, 16);

                var points = new PointF[4]
                {
                    new PointF(shape.Left, shape.Top),
                    new PointF(shape.Right, shape.Top),
                    new PointF(shape.Right, shape.Bottom),
                    new PointF(shape.Left, shape.Bottom),
                };

                image.Mutate(q => q.FillPolygon(Rgba32.White, points));
            }
        }


        public static bool[] CreateTarget(byte value, int numberOfClasses)
        {
            var res = new bool[numberOfClasses];
            res[value] = true;
            return res;
        }

        public static double[,,] Create3DInput(byte[] values)
        {
            int size = (int)Math.Sqrt(values.Length);

            var output = new double[1, size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    output[0, i, j] = (double)values[i * size + j] / 255;
                }
            }

            return output;
        }

        public static double[,,] Create3DInput(double[,] values)
        {
            int size = (int)Math.Sqrt(values.Length);

            var output = new double[1, size, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    output[0, i, j] = values[i,j];
                }
            }

            return output;
        }

        public static double[] Create1DInput(byte[] values)
        {
            return values.Select((q, i) => (double)q / 255).ToArray();
        }

        public static byte IntegerFromOutput(double[] values)
        {
            return (byte)(values.ToList().IndexOf(values.Max()));
        }
    }
}
