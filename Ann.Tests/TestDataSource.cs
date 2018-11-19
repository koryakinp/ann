using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;

namespace Ann.Tests
{
    class TestDataSource : Attribute, ITestDataSource
    {

        public readonly int Min;
        public readonly int Max;

        public TestDataSource(int min, int max)
        {
            Min = min;
            Max = max;
        }

        public IEnumerable<object[]> GetData(MethodInfo methodInfo)
        {
            return Enumerable.Range(Min, Max).Select(q => new object[] { q });
        }

        public string GetDisplayName(MethodInfo methodInfo, object[] data)
        {
            if (data != null)
            {
                return string.Format(CultureInfo.CurrentCulture, "{0} ({1})", methodInfo.Name, string.Join(",", data));
            }
            return null;
        }
    }
}
