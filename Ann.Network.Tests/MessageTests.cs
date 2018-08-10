using Ann.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Ann.Network.Tests
{
    [TestClass]
    public class MessageTests
    {
        private readonly double[,,] _multi;
        private readonly double[] _single;

        public MessageTests()
        {
            _single = new double[18] { 4,3,6,2,7,1,3,5,6,5,7,8,3,6,8,3,9,2 };

            _multi = new double[2, 3, 3]
            {
                {
                    { 4,3,6 },
                    { 2,7,1 },
                    { 3,5,6 }
                },
                {
                    { 5,7,8 },
                    { 3,6,8 },
                    { 3,9,2 }
                }
            };
        }

        [TestMethod]
        public void ToMultiTest()
        {
            var message = new Message(_single);
            var actual = message.ToMulti(_multi.GetLength(0), _multi.GetLength(1));

            CollectionAssert.AreEqual(actual, _multi);
        }

        [TestMethod]
        public void ToSingleTest()
        {
            var message = new Message(_multi);
            var actual = message.ToSingle();

            CollectionAssert.AreEqual(actual, _single);
        }

        [TestMethod]
        public void ConvertTest()
        {
            var message1 = new Message(_multi);
            CollectionAssert.AreEqual(message1.ToMulti(_multi.GetLength(0), _multi.GetLength(1)), _multi);

            var message2 = new Message(_single);
            CollectionAssert.AreEqual(message2.ToSingle(), _multi);

            var message3 = new Message(_multi);
            var multi = new Message(message3.ToSingle()).ToMulti(_multi.GetLength(0), _multi.GetLength(1));
            CollectionAssert.AreEqual(multi, _multi);

            var message4 = new Message(_single);
            var single = new Message(message4.ToMulti(_multi.GetLength(0), _multi.GetLength(1))).ToSingle();
            CollectionAssert.AreEqual(single, _single);
        }
    }
}
