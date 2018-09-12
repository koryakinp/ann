using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ann.Utils
{
    public static class ArrayExtensions
    {
        private static void ForEachh<T>(this Array array, Func<T, T> action)
        {
            var dimensionSizes = Enumerable
                .Range(0, array.Rank)
                .Select(i => array.GetLength(i))
                .ToArray();

            ArrayForEach(dimensionSizes, action, new int[0], array);
        }

        private static void ForEachh<T>(this Array array, Func<T, int[], T> action)
        {
            var dimensionSizes = Enumerable
                .Range(0, array.Rank)
                .Select(i => array.GetLength(i))
                .ToArray();

            ArrayForEach(dimensionSizes, action, new int[0], array);
        }

        private static void ArrayForEach<T>(int[] dimensionSizes, Func<T, T> action, int[] externalCoordinates, Array masterArray)
        {
            if (dimensionSizes.Length == 1)
            {
                for (int i = 0; i < dimensionSizes[0]; i++)
                {
                    var globalCoordinates = externalCoordinates.Concat(new[] { i }).ToArray();
                    var value = (T)masterArray.GetValue(globalCoordinates);
                    masterArray.SetValue(action(value), globalCoordinates);
                }
            }
            else
            {
                for (int i = 0; i < dimensionSizes[0]; i++)
                {
                    ArrayForEach(dimensionSizes.Skip(1).ToArray(), action, externalCoordinates.Concat(new[] { i }).ToArray(), masterArray);
                }
            }
        }

        private static void ArrayForEach<T>(int[] dimensionSizes, Func<T,int[],T> action, int[] externalCoordinates, Array masterArray)
        {
            if (dimensionSizes.Length == 1)
            {
                for (int i = 0; i < dimensionSizes[0]; i++)
                {
                    var globalCoordinates = externalCoordinates.Concat(new[] { i }).ToArray();
                    var value = (T)masterArray.GetValue(globalCoordinates);
                    masterArray.SetValue(action(value, globalCoordinates), globalCoordinates);
                }
            }
            else
            {
                for (int i = 0; i < dimensionSizes[0]; i++)
                {
                    ArrayForEach(dimensionSizes.Skip(1).ToArray(), action, externalCoordinates.Concat(new[] { i }).ToArray(), masterArray);
                }
            }
        }

        public static void UpdateForEach<T>(this Array array, Func<T,T> calculateElement)
        {
            array.ForEachh<T>((element) => calculateElement(element));
        }

        public static void UpdateForEach<T>(this Array array, Func<T,int[],T> calculateElement)
        {
            array.ForEachh<T>((element, idx) => calculateElement(element, idx));
        }
    }
}
