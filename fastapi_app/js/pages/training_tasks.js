var currentTask = 1;
var totalTasks = 5;

function showTask(taskNumber) {
    // Hide all tasks
    for (var i = 1; i <= totalTasks; i++) {
        document.getElementById('task' + i).classList.remove('active');
        document.getElementById('wizElement' + i).classList.remove('active');

        // Collapse all accordion items in this task
        var taskAccordionItems = document.querySelectorAll('#task' + i + ' .accordion-collapse');
        taskAccordionItems.forEach(function (item) {
            item.classList.remove('show');
        });
    }
    // Show the selected task
    document.getElementById('task' + taskNumber).classList.add('active');
    document.getElementById('wizElement' + taskNumber).classList.add('active');
    currentTask = taskNumber;

    // Expand the first accordion item in the selected task
    var firstAccordionItem = document.querySelector('#task' + taskNumber + ' .accordion .accordion-item:first-child .accordion-collapse');
    if (firstAccordionItem) {
        firstAccordionItem.classList.add('show');
    }

    // Update Previous and Next buttons
    updateNavigationButtons();
}

function nextTask() {
    if (currentTask < totalTasks) {
        showTask(currentTask + 1);
    }
}

function prevTask() {
    if (currentTask > 1) {
        showTask(currentTask - 1);
    }
}

function updateNavigationButtons() {
    // Disable 'Previous' button on first task
    if (currentTask === 1) {
        document.getElementById('prevButton').disabled = true;
    } else {
        document.getElementById('prevButton').disabled = false;
    }

    // Disable 'Next' button on last task
    if (currentTask === totalTasks) {
        document.getElementById('nextButton').disabled = true;
    } else {
        document.getElementById('nextButton').disabled = false;
    }
}

// Initialize navigation buttons and expand first accordion item of the first task
document.addEventListener('DOMContentLoaded', function () {
    updateNavigationButtons();
    showTask(currentTask);
});
