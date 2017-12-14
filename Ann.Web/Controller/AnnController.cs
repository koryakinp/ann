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

        private readonly RecognitionResult _rr;

        public AnnController(RecognitionResult rr)
        {
            _rr = rr;
        }

        [HttpGet]
        [Route("ann/recognize")]
        public RecognitionResult Recognize(string data)
        {
            return _rr;
        }
    }
}