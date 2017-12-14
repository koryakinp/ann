/// <binding BeforeBuild='copy-assets' />
"use strict";

var _ = require('lodash'),
    gulp = require('gulp');

gulp.task('copy-assets', function () {
    var assets = {
        js: [
            './node_modules/jquery/dist/jquery.min.js',
        ]
    };
    _(assets).forEach(function (assets, type) {
        gulp.src(assets).pipe(gulp.dest('./wwwroot/lib/' + type));
    });
});