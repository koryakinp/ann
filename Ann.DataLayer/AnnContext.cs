using Ann.DataLayer.Entities;
using Microsoft.EntityFrameworkCore;

namespace Ann.DataLayer
{
    public class AnnContext : DbContext
    {
        public AnnContext(DbContextOptions<AnnContext> options) : base(options) {}

        public DbSet<Lesson> Lessons { get; set; }
        public DbSet<LayerConfiguration> LayerConfiguration { get; set; }
    }
}
