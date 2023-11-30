function stopVideo() {
    var video = document.getElementById("tutorialVideo");
    video.pause();
}

function change_shs_box_visibility() {
    if (document.getElementById("selectShs").checked) {
        document.getElementById('selectShsBox').classList.remove('box--not-selected');
        document.getElementById('shs_max_grid_cost').disabled = false;
        document.getElementById('lblShsLifetime').classList.remove('disabled');
        document.getElementById('shsLifetimeUnit').classList.remove('disabled');
        if (document.getElementById('shs_max_grid_cost').value === '') {
            document.getElementById('shs_max_grid_cost').value = '0.6';
        }
    } else {
        document.getElementById('selectShsBox').classList.add('box--not-selected');
        document.getElementById('shs_max_grid_cost').disabled = true;
        document.getElementById('lblShsLifetime').classList.add('disabled');
        document.getElementById('shsLifetimeUnit').classList.add('disabled');
    }
}
