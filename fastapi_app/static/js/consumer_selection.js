/************************************************************/
/*                ACTIVATION / INACTIVATION                 */
/************************************************************/
function activation_check() {
    if (document.getElementById('selectionFile').checked) {
        document.getElementById('fileImport').disabled = false
        document.getElementById('btnImport').classList.remove('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.add('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.add('disabled');
    } else if (document.getElementById('selectionBoundaries').checked) {
        document.getElementById('fileImport').disabled = true
        document.getElementById('btnImport').classList.add('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.remove('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.remove('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.remove('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.remove('disabled');
    } else if (document.getElementById('selectionMap').checked) {
        document.getElementById('fileImport').disabled = true
        document.getElementById('btnImport').classList.add('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.add('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.add('disabled');
    }
}