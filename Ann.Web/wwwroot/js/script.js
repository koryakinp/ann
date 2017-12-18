var canvas = new Canvas();
canvas.reset();

$('#recognize').click(sendImage);
$('#clear').click(function () {
    canvas.reset();
    $('#result').text('');
});

function sendImage() {
    $.ajax({
        type: "POST",
        url: "ann/recognize",
        data: canvas.getImageData()
    }).done(function (o) {
        $('#result').text('I am ' + o.confidence.toFixed(2)
            + '% confident, that the number you draw was '
            + o.number);
    });
}