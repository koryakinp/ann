using Ann.DataLayer;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.IO;

namespace Ann.Learning
{
    class Program
    {
        private static IConfigurationRoot Configuration;

        static void Main(string[] args)
        {
            var services = new ServiceCollection();

            var builder = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", false)
                .AddEnvironmentVariables();

            Configuration = builder.Build();

            services
                .AddDbContext<AnnContext>(q => q.UseSqlServer(Configuration.GetConnectionString("Default")));

            services.BuildServiceProvider();
        }
    }
}
