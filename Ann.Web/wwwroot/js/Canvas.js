function Canvas() {

    var that = this;
    var lines = [];

    this.reset = function () {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        lines.length = 0;
    }

    this.getImageData = function() {
        return {
            data: el.toDataURL("image/jpeg")
        };
    }

    function midPointBtw(p1, p2) {
        return {
            x: p1.x + (p2.x - p1.x) / 2,
            y: p1.y + (p2.y - p1.y) / 2
        };
    }

    var el = document.getElementById('canvas');
    var ctx = el.getContext('2d');

    ctx.lineWidth = 5;
    ctx.lineJoin = ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    var isDrawing, points = [];

    $('#canvas').mousedown(function (e) {
        isDrawing = true;
        lines.push([]);
        lines[lines.length - 1].push({ x: e.pageX - this.offsetLeft, y: e.pageY - this.offsetTop });

    });

    $('#canvas').mousemove(function (e) {
        if (!isDrawing) return;

        lines[lines.length - 1].push({ x: e.pageX - this.offsetLeft, y: e.pageY - this.offsetTop });

        for (var j = 0; j < lines.length; j++) {
            var points = lines[j];

            var p1 = points[0];
            var p2 = points[1];

            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);

            for (var i = 1, len = points.length; i < len; i++) {
                var midPoint = midPointBtw(p1, p2);
                ctx.quadraticCurveTo(p1.x, p1.y, midPoint.x, midPoint.y);
                p1 = points[i];
                p2 = points[i + 1];
            }

            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
        }
        
    });

    $('#canvas').mouseup(function (e) {
        isDrawing = false;
    });
}