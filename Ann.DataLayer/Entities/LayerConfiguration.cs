using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Ann.DataLayer.Entities
{
    public class LayerConfiguration
    {
        [Key]
        public Guid Id { get; set; }
        public int Order { get; set; }
        public int NumberOfNeurons { get; set; }
        public ActivatorType Activator { get; set; }

        public int LessonId { get; set; }
        [ForeignKey("LessonId")]
        public virtual Guid Lesson { get; set; }
    }
}
