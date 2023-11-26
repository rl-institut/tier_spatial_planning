function show_modal_example_model() {
    // Select the table by its ID 'projectTable'
    var table = document.getElementById('projectTable');

    // The table rows, excluding the header
    var rows = table.querySelectorAll('tr:not(:first-child)');

    // If there are no rows (excluding the header), it means there are no projects
    if (rows.length == 1 && rows[0].innerText.includes("You do not yet have any saved projects")) {
        document.getElementById('projectExample').style.cssText = "display: block !important;";
    }
}

function set_and_show_error_msg() {
    if(window.location.href.includes("/?internal_error")) {
        let message = "An internal server error occurred!";
        if(navigator.userAgent.indexOf("Firefox") !== -1){
            message = "An internal server error occurred!\nIt appears that you are using the Firefox browser. " +
                "The error could be related to Firefox. We recommend switching to Chrome or Edge instead.";}
        document.getElementById('responseMsg').textContent = message;
        document.getElementById('msgBox').style.display = "block";
    }
}
