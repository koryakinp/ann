﻿using Ann.Layers.Flatten;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Ann.Core.Tests.FlattenLayer
{
    [TestClass]
    public class FlattenLayerTests
    {
        private FlattenFullLayer _layer { get; set; }

        [TestInitialize]
        public void Initialize()
        {
            _layer = new FlattenFullLayer(new MessageShape(5, 3));
        }

        [TestMethod]
        public void ForwardPassTest()
        {
            var actual = _layer.PassForward(FlattenLayerTestsData.ForwardPassInput);
            var expected = FlattenLayerTestsData.BackwardPassInput;
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void BackwardPassTest()
        {
            var actual = _layer.PassBackward(FlattenLayerTestsData.BackwardPassInput);
            var expected = FlattenLayerTestsData.ForwardPassInput;
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
