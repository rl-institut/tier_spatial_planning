var currentDate = new Date();
var year = currentDate.getFullYear();

// Select the element to be observed
const responseMsgElement = document.getElementById('responseMsg');

// Create a new MutationObserver instance
const observer = new MutationObserver(function(mutationsList, observer) {
    for (const mutation of mutationsList) {
        if (mutation.type === 'characterData' || mutation.type === 'childList') {
            // Show the modal when textContent of responseMsg changes
            document.getElementById('msgBox').style.display = 'block';
        }
    }
});

// Define what to observe (changes to the text content of the element)
observer.observe(responseMsgElement, {
    characterData: true,  // Observes changes to the text content
    childList: true,      // Observes addition/removal of child nodes
    subtree: true         // Observes changes within the descendants of the node
});


// Function to update the visibility of the Grid Design step
function updateGridDesignVisibility() {
    const gridDesignStep = document.querySelector('li[onclick*="grid_design"]');
    const toggleSwitch1 = document.getElementById('toggleswitch1');

    if (toggleSwitch1.checked) {
        gridDesignStep.style.display = ''; // Show Grid Design step if activated
    } else {
        gridDesignStep.style.display = 'none'; // Hide Grid Design step if deactivated
    }
}

// Function to update the visibility of the Energy System Design step
function updateEnergySystemDesignVisibility() {
    const energySystemDesignStep = document.querySelector('li[onclick*="energy_system_design"]');
    const toggleSwitch2 = document.getElementById('toggleswitch2');

    if (toggleSwitch2.checked) {
        energySystemDesignStep.style.display = ''; // Show Energy System Design step if activated
    } else {
        energySystemDesignStep.style.display = 'none'; // Hide Energy System Design step if deactivated
    }
}

// Function to update the visibility of the Consumer Selection step
function updateConsumerSelectionVisibility() {
    const consumerSelectionStep = document.querySelector('li[onclick*="consumer_selection"]');
    const toggleSwitch0 = document.getElementById('toggleswitch0');
    const toggleSwitch1 = document.getElementById('toggleswitch1');

    if (!toggleSwitch0.checked && !toggleSwitch1.checked) {
        consumerSelectionStep.style.display = 'none'; // Hide if both are deactivated
    } else {
        consumerSelectionStep.style.display = ''; // Show if either is activated
    }
}

// Function to set the correct href for the Next button based on the visibility of wizard steps
function updateNextButtonHref(project_id) {
    const consumerSelectionStep = document.querySelector('li[onclick*="consumer_selection"]');
    const nextButton = document.getElementById("nextButton");

    let nextHref = 'consumer_selection?project_id=' + project_id;

    if (consumerSelectionStep.style.display === 'none') {
        // If Consumer Selection is hidden, change the next step to Demand Estimation
        nextHref = 'demand_estimation?project_id=' + project_id;
    }

    // Update the onclick attribute of the Next button
    nextButton.setAttribute('onclick', `save_project_setup(${project_id}, '${nextHref}');`);
}

// Call this function whenever the visibility of the wizard steps might change
document.getElementById('toggleswitch0').addEventListener('change', function() {
    updateConsumerSelectionVisibility(); // Update Consumer Selection visibility
    updateNextButtonHref(0); // Update the Next button's href based on the project ID
});

document.getElementById('toggleswitch1').addEventListener('change', function() {
    updateGridDesignVisibility();           // Update Grid Design visibility based on toggleswitch1
    updateEnergySystemDesignVisibility();   // Update Energy System Design visibility based on toggleswitch1
    updateConsumerSelectionVisibility();    // Update Consumer Selection visibility
    updateNextButtonHref(0); // Update the Next button's href based on the project ID
});

document.getElementById('toggleswitch2').addEventListener('change', function() {
    updateEnergySystemDesignVisibility();   // Update Energy System Design visibility based on toggleswitch2
    updateNextButtonHref(0); // Update the Next button's href based on the project ID
});

// Initial call to set the correct href on page load
window.addEventListener('load', function() {
    updateNextButtonHref(project_id); // Update the Next button's href based on the project ID
});
