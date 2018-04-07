using System;
using System.Collections.Generic;
using System.Text;

namespace Ann
{
    public static class Extensions
    {
        public static IEnumerable<T> FastReverse<T>(this IList<T> items)
        {
            for (int i = items.Count - 1; i >= 0; i--)
            {
                yield return items[i];
            }
        }

        public static void ForEach<T>(this IEnumerable<T> source, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in source) action(e, i++);
        }

        public static void ForEach<T>(this IEnumerable<T> source, Action<T> action)
        {
            foreach (T element in source)
                action(element);
        }

        public static void ForEach<T>(this T[,] source, Action<T, int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(source[w, h], w, h);
                }
            }
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
}
