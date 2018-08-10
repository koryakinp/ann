using System;

namespace Ann.Core
{
    public class Message
    {
        internal readonly Array Value;

        public Message(double[] value)
        {
            Value = value;
        }

        public Message(double[,,] value)
        {
            Value = value;
        }

        public double[] ToSingle()
        {
            if(!IsMulti)
            {
                return Value as double[];
            }

            if (Value.Length == 0)
            {
                throw new Exception(Consts.CanNotConvertMessage);
            }

            double[] output = new double[Value.Length];

            int depth = Value.GetLength(0);
            int width = Value.GetLength(1);
            int height = Value.GetLength(2);

            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        output[i * width * height + j * width + k] = (double)Value.GetValue(i, j, k);
                    }
                }
            }

            return output;
        }

        public double[,,] ToMulti(int channels, int size)
        {
            var output = new double[channels, size, size];

            if(IsMulti)
            {
                if(Value.GetLength(0) != channels ||
                    Value.GetLength(1) != size && Value.GetLength(2) != size)
                {
                    throw new Exception(Consts.CanNotConvertMessage);
                }

                return Value as double[,,];
            }


            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < size; j++)
                {
                    for (int k = 0; k < size; k++)
                    {
                        output[i, j, k] = (double)Value.GetValue(i * size * size + j * size + k);
                    }
                }
            }

            return output;
        }

        internal bool IsMulti => Value.Rank > 1;
    }
}