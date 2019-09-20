async function app() {
    const url = prompt();
    document.getElementById('img').src = url;
    const img = document.getElementById('img');

    const net = await mobilenet.load();

    const pred = await net.classify(img);

    console.log(pred);

    document.getElementById('pred').innerHTML = pred[0].className;
}

app();