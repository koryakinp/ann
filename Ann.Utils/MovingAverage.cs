using System.Collections.Generic;
using System.Linq;

namespace Ann.Utils
{
    public class MovingAverage
    {
        private readonly Queue<double> _queue;
        private readonly int _period;

        public MovingAverage(int period)
        {
            _period = period;
            _queue = new Queue<double>(period);
        }

        public double Compute(double x)
        {
            if (_queue.Count >= _period)
            {
                _queue.Dequeue();
            }
            else
            {
                _queue.Enqueue(x);
            }

            return _queue.Average();
        }
    }
}
