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


const consumerSelectionHref = `consumer_selection?project_id=${project_id}`;
const demandEstimationHref = `demand_estimation?project_id=${project_id}`;
const func = `save_project_setup`;

// Function to handle toggleswitch0 (Demand Estimation)
document.getElementById('toggleswitch0').addEventListener('change', function() {
    if (!this.checked) {
        // Update the text content of the responseMsg element
        document.getElementById('responseMsg').textContent =
            "If you keep the demand estimation option enabled, you can choose later between estimating demand or using a custom demand time series. If the demand estimation function is not used, a corresponding time series must be uploaded later in the 'Demand Estimation' section.";
        // Show the modal
        document.getElementById('msgBox').style.display = 'block';
    }
});

// Function to handle toggleswitch1 (Spatial Grid Optimization)
document.getElementById('toggleswitch1').addEventListener('change', function() {
    if (!this.checked) {
        // Update the text content of the responseMsg element
        document.getElementById('responseMsg').textContent =
            "A demand is required for the design optimization of energy converters. Demand estimation requires information about consumers, which is defined in the 'Consumer Selection' section using the integrated mapping system. Therefore, even if grid planning is not carried out, consumers must still be specified unless a custom demand time series is uploaded in the 'Demand Estimation' section. In that case, also deactivate 'Demand Estimation' to skip the consumer definition step.";

        // Show the modal
        document.getElementById('msgBox').style.display = 'block';
    }
});


// Function to set the correct href for the Next button based on the visibility of wizard steps
function updateNextButtonHref(project_id, func, defaultHref, alternativeHref) {
    const consumerSelectionStep = document.querySelector('li[onclick*="consumer_selection"]');
    const nextButton = document.getElementById("nextButton");
    if (consumerSelectionStep.style.display === 'none') {
        // If Consumer Selection is hidden, use the alternative href
        nextButton.setAttribute('onclick', `${func}('${alternativeHref}');`);
    } else {
        // If Consumer Selection is visible, use the default href
        nextButton.setAttribute('onclick', `${func}('${defaultHref}');`);
    }
}


// Event listeners passing the states of toggleswitches to the function
document.getElementById('toggleswitch0').addEventListener('change', function() {
    const toggleSwitch0State = this.checked;
    const toggleSwitch1State = document.getElementById('toggleswitch1').checked;
    const toggleSwitch2State = document.getElementById('toggleswitch2').checked;
    updateWizardStepVisibility(toggleSwitch0State, toggleSwitch1State, toggleSwitch2State);
    updateNextButtonHref(project_id, func, consumerSelectionHref, demandEstimationHref);
});

document.getElementById('toggleswitch1').addEventListener('change', function() {
    const toggleSwitch0State = document.getElementById('toggleswitch0').checked;
    const toggleSwitch1State = this.checked;
    const toggleSwitch2State = document.getElementById('toggleswitch2').checked;

    updateWizardStepVisibility(toggleSwitch0State, toggleSwitch1State, toggleSwitch2State);
    updateNextButtonHref(project_id, func, consumerSelectionHref, demandEstimationHref);
});

document.getElementById('toggleswitch2').addEventListener('change', function() {
    const toggleSwitch0State = document.getElementById('toggleswitch0').checked;
    const toggleSwitch1State = document.getElementById('toggleswitch1').checked;
    const toggleSwitch2State = this.checked;

    updateWizardStepVisibility(toggleSwitch0State, toggleSwitch1State, toggleSwitch2State);
    updateNextButtonHref(project_id, func, consumerSelectionHref, demandEstimationHref);
});


const wizardSection = document.getElementById('wizard');
wizardSection.classList.add('show');
