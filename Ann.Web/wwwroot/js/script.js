var ctx = document.getElementById('canvas').getContext("2d");

$('#canvas').mousedown(function (e) {
    var mouseX = e.pageX - this.offsetLeft;
    var mouseY = e.pageY - this.offsetTop;

    paint = true;
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
    redraw();
});

$('#recognize').click(function(){
    sendImage();
});

$('#clear').click(function () {
    reset();
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
});

$('#canvas').mousemove(function (e) {
    if (paint) {
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
        redraw();
    }
});

$('#canvas').mouseup(function (e) {
    paint = false;
});

$('#canvas').mouseleave(function (e) {
    paint = false;
});

function reset() {
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
}

var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint;

function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

function redraw() {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clears the canvas
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    ctx.strokeStyle = 'white';
    ctx.lineJoin = "round";
    ctx.lineWidth = 5;

    for (var i = 0; i < clickX.length; i++) {
        ctx.beginPath();
        if (clickDrag[i] && i) {
            ctx.moveTo(clickX[i - 1], clickY[i - 1]);
        } else {
            ctx.moveTo(clickX[i] - 1, clickY[i]);
        }
        ctx.lineTo(clickX[i], clickY[i]);
        ctx.closePath();
        ctx.stroke();
    }
}

function sendImage() {
    var data = document
        .getElementById('canvas')
        .toDataURL("image/jpeg");

    $.ajax({
        type: "POST",
        url: "ann/recognize",
        data: {
            data: data
        }
    }).done(function (o) {
        console.log('saved');
    });
}