using Ann.DataLayer.Enums;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Ann.DataLayer.Entities
{
    public class Lesson
    {
        [Key]
        public Guid Id { get; set; }

        public double SuccessRatio { get; set; }
        public double LearningRate { get; set; }
        public double LearningRateDecay { get; set; }
        public int LearningRateDecayPeriod { get; set; }
        public double Momentum { get; set; }
        public LearningState State { get; set; }
        public LearningRateDecayType LearningRateDecayType { get; set; }

        public virtual ICollection<LayerConfiguration> LayerConfigurations { get; set; }

    }
}
