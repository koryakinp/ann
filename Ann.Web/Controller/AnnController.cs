using System;
using Microsoft.AspNetCore.Mvc;
using Ann.Web.Models;
using ImageSharp;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using ImageSharp.Processing.Processors;

namespace Ann.Web.Controller
{
    [Produces("application/json")]
    public class AnnController
    {
        private readonly Network _network;

        public AnnController(Network network)
        {
            _network = network;
        }

        [HttpPost]
        [Route("ann/recognize")]
        public RecognitionResult Recognize(string data)
        {
            Image<Color> img = DecodeStringToImage(data);
            Image<Color> centered = CenterImage(img);
            List<byte> bytes = ConverImageToBytes(centered);

            var result = _network.UseModel(bytes.Select(q => (double)q/255).ToList());

            double confidence = (result.Max() / result.Sum())*100;
            int number = result.IndexOf(result.Max());

            return new RecognitionResult
            {
                Confidence = confidence,
                Number = number
            };
        }

        private Image<Color> DecodeStringToImage(string data)
        {
            data = data.Replace("data:image/jpeg;base64,", "");
            var img = new Image<Color>(Convert.FromBase64String(data));
            img.Resize(28, 28);
            return img;
        }

        private Rectangle GetBoundingRect(Image<Color> img)
        {
            int left = 28;
            int right = 0;
            int top = 28;
            int bot = 0;

            for (int i = 0; i < img.Width; i++)
            {
                for (int j = 0; j < img.Height; j++)
                {
                    if(img.Pixels[i * img.Width + j].R != 0 && i < left)
                    {
                        left = i;
                    }

                    if (img.Pixels[i * img.Width + j].R != 0 && i > right)
                    {
                        right = i;
                    }

                    if (img.Pixels[i * img.Width + j].R != 0 && j < top)
                    {
                        top = j;
                    }

                    if (img.Pixels[i * img.Width + j].R != 0 && j > bot)
                    {
                        bot = j;
                    }
                }
            }

            return new Rectangle(new Point(top, left), new Point(bot, right));
        }

        private Image<Color> CenterImage(Image<Color> img)
        {
            Rectangle rect = GetBoundingRect(img);
            return img.Crop(rect).Pad(28, 28);
        }

        private List<byte> ConverImageToBytes(Image<Color> img)
        {
            List<byte> bytes = new List<byte>();
            for (int i = 0; i < img.Width; i++)
            {
                for (int j = 0; j < img.Height; j++)
                {
                    bytes.Add(img.Pixels[i * img.Width + j].R);
                }
            }
            return bytes;
        }
    }
}