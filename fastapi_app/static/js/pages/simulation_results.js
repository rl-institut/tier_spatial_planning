document.getElementById('downloadCSV').addEventListener('click', function() {
window.location.href = '/download_data/{{ project_id }}/csv';
        });
var targetNode = document.getElementById('responseMsg');
var config = { childList: true, subtree: true, characterData: true };
var callback = function(mutationsList, observer) {
    for(let mutation of mutationsList) {
        if ((mutation.type === 'childList' || mutation.type === 'characterData') && targetNode.textContent.trim() !== '') {
            var modal = document.getElementById('msgBox');
            modal.style.display = "block";
        }
    }
};
var observer = new MutationObserver(callback);
observer.observe(targetNode, config);
document.getElementById("msgBox").style.zIndex = "9999";

