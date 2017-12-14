using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Ann.Web.Models;

namespace Ann.Web.Controller
{
    [Produces("application/json")]
    public class AnnController
    {
        [HttpGet]
        [Route("ann/recognize")]
        public RecognitionResult Recognize(string data)
        {
            return new RecognitionResult
            {
                Confidence = 10,
                Number = 2
            };
        }
    }
}