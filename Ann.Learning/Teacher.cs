using Ann.Configuration;
using Ann.DataLayer;
using Ann.DataLayer.Entities;
using Ann.DataLayer.Enums;
using System.Collections.Generic;
using System.Linq;
using NetworkLayerConfiguration = Ann.Configuration.LayerConfiguration;
using LearningLayerConfiguration = Ann.DataLayer.Entities.LayerConfiguration;
using System;
using Ann.Decayers;
using Ann.Utils;

namespace Ann.Learning
{
    public class Teacher
    {
        private readonly AnnContext _ctx;

        public Teacher(AnnContext ctx)
        {
            _ctx = ctx;
        }

        public List<Lesson> GetLessons(int lessons)
        {
            return _ctx.Lessons
                .Where(q => q.State == LearningState.NotStarted)
                .Take(lessons)
                .ToList();
        }

        public void Teach(List<Lesson> lessons)
        {
            foreach (var lesson in lessons)
            {
                Network n = BuildNetwork(lesson);
                List<Image> images = ImageProvider.ProvideImages(@"minst/train.zip");
                foreach (var image in images)
                {
                    n.TrainModel(image.Data.Select(q => (double)q/255).ToList(), NetworkHelper.CreateTarget(image.Value));
                }
            }
        }

        private Network BuildNetwork(Lesson lesson)
        {
            NetworkLayerConfiguration lcfg = BuildLayerConfiguration(lesson.LayerConfigurations.ToList());
            NetworkConfiguration ncfg = new NetworkConfiguration(lcfg);

            ncfg.Momentum = lesson.Momentum;
            ncfg.LearningRateDecayer = BuildLearningRateDecayer(lesson);
            ncfg.LearningRate = lesson.LearningRate;

            throw new NotImplementedException();
        }

        private NetworkLayerConfiguration BuildLayerConfiguration(List<LearningLayerConfiguration> lcfgList)
        {
            NetworkLayerConfiguration lcfg = new NetworkLayerConfiguration();
            foreach (var lc in lcfgList.OrderBy(q => q.Order))
            {
                if (lc.Order == 1)
                {
                    lcfg.AddInputLayer(lc.NumberOfNeurons);
                }
                else if (lc.Order == lcfgList.Max(q => q.Order))
                {
                    lcfg.AddOutputLayer(lc.NumberOfNeurons, lc.Activator);
                }
                else
                {
                    lcfg.AddHiddenLayer(lc.NumberOfNeurons, lc.Activator);
                }
            }

            return lcfg;
        }

        private ILearningRateDecayer BuildLearningRateDecayer(Lesson lesson)
        {
            switch (lesson.LearningRateDecayType)
            {
                case LearningRateDecayType.Exponential:
                    return new ExponentialDecayer(lesson.LearningRate, lesson.LearningRateDecay);
                case LearningRateDecayType.Step:
                    return new StepDecayer(lesson.LearningRate, lesson.LearningRateDecay, lesson.LearningRateDecayPeriod);
                default: return null;
            }
        }
    }
}
