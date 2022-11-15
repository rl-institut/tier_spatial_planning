function boxVisibilityShs() {
    // ---------- ENABLING ITEMS ---------- //
    const tierLevels = ["One", "Two", "Three", "Four", "Five"];

    if (document.getElementById("selectShs").checked) {
        // Change the border color.
        document.getElementById("selectShsBox").classList.remove('box--not-selected');

        for (tierLevel in tierLevels) {
            document.getElementById("shsTier"+tierLevels[tierLevel]+"Cost").disabled = false;
            document.getElementById("lblShsTier"+tierLevels[tierLevel]+"Cost").classList.remove('disabled');
            document.getElementById("icnShsTier"+tierLevels[tierLevel]+"Tooltip").classList.remove('disabled');
            document.getElementById("shsTier"+tierLevels[tierLevel]+"CostUnit").classList.remove('disabled');
        }
    } else {
        // Change the border color.
        document.getElementById("selectShsBox").classList.add('box--not-selected');
        
        for (tierLevel in tierLevels) {
            document.getElementById("shsTier"+tierLevels[tierLevel]+"Cost").disabled = true;
            document.getElementById("lblShsTier"+tierLevels[tierLevel]+"Cost").classList.add('disabled');
            document.getElementById("icnShsTier"+tierLevels[tierLevel]+"Tooltip").classList.add('disabled');
            document.getElementById("shsTier"+tierLevels[tierLevel]+"CostUnit").classList.add('disabled');
        }
    }
}