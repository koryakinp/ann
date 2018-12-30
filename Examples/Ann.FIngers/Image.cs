namespace Ann.Fingers
{
    public class Image
    {
        public readonly double[,] Data;
        public readonly byte Label;
        public readonly byte[] ByteData;

        public Image(byte[] data, byte label)
        {
            Data = new double[128, 128];

            for (int i = 0; i < 128; i++)
            {
                for (int j = 0; j < 128; j++)
                {
                    Data[i, j] = (double)data[i * 128 + j]/255;
                }
            }

            Label = label;
            ByteData = data;
        }
    }
}
