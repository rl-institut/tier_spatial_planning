function boxVisibilityShs() {
    // ---------- ENABLING ITEMS ---------- //
    const tierLevels = ["One", "Two", "Three", "Four", "Five"];

    if (document.getElementById("selectShs").checked) {
        // Change the border color.
        document.getElementById("selectShsBox").classList.remove('box--not-selected');

        for (tierLevel in tierLevels) {
            document.getElementById("shsLifetime").disabled = false;
            document.getElementById("lblShsLifetime").classList.remove('disabled');
            document.getElementById("shsLifetimeUnit").classList.remove('disabled');
            document.getElementById("shsTier"+tierLevels[tierLevel]+"Capex").disabled = false;
            document.getElementById("lblShsTier"+tierLevels[tierLevel]+"Capex").classList.remove('disabled');
            document.getElementById("icnShsTier"+tierLevels[tierLevel]+"Tooltip").classList.remove('disabled');
            document.getElementById("shsTier"+tierLevels[tierLevel]+"CapexUnit").classList.remove('disabled');
        }
    } else {
        // Change the border color.
        document.getElementById("selectShsBox").classList.add('box--not-selected');
        
        for (tierLevel in tierLevels) {
            document.getElementById("shsLifetime").disabled = true;
            document.getElementById("lblShsLifetime").classList.add('disabled');
            document.getElementById("shsLifetimeUnit").classList.add('disabled');
            document.getElementById("shsTier"+tierLevels[tierLevel]+"Capex").disabled = true;
            document.getElementById("lblShsTier"+tierLevels[tierLevel]+"Capex").classList.add('disabled');
            document.getElementById("icnShsTier"+tierLevels[tierLevel]+"Tooltip").classList.add('disabled');
            document.getElementById("shsTier"+tierLevels[tierLevel]+"CapexUnit").classList.add('disabled');
        }
    }
}