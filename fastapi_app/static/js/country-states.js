// All Country and region name
// Code generated by https://www.html-code-generator.com/html/drop-down/country-region

// country names and code

var country_and_states = {
country: {"NG":"Nigeria"},


 //states name and code

states: {"NG":[{"name":"Abia State","code":"AB"},{"name":"Adamawa State","code":"AD"},{"name":"Akwa Ibom State","code":"AK"},{"name":"Anambra State","code":"AN"},{"name":"Bauchi State","code":"BA"},{"name":"Bayelsa State","code":"BY"},{"name":"Benue State","code":"BE"},{"name":"Borno State","code":"BO"},{"name":"Cross River State","code":"CR"},{"name":"Delta State","code":"DE"},{"name":"Ebonyi State","code":"EB"},{"name":"Edo State","code":"ED"},{"name":"Ekiti State","code":"EK"},{"name":"Enugu State","code":"EN"},{"name":"Federal Capital Territory","code":"FC"},{"name":"Gombe State","code":"GO"},{"name":"Imo State","code":"IM"},{"name":"Jigawa State","code":"JI"},{"name":"Kaduna State","code":"KD"},{"name":"Kano State","code":"KN"},{"name":"Katsina State","code":"KT"},{"name":"Kebbi State","code":"KE"},{"name":"Kogi State","code":"KO"},{"name":"Kwara State","code":"KW"},{"name":"Lagos","code":"LA"},{"name":"Nasarawa State","code":"NA"},{"name":"Niger State","code":"NI"},{"name":"Ogun State","code":"OG"},{"name":"Ondo State","code":"ON"},{"name":"Osun State","code":"OS"},{"name":"Oyo State","code":"OY"},{"name":"Plateau State","code":"PL"},{"name":"Sokoto State","code":"SO"},{"name":"Taraba State","code":"TA"},{"name":"Yobe State","code":"YO"},{"name":"Zamfara State","code":"ZA"}],}
};

// user country code for selected option
let user_country_code = "NG";

(function () {
    // script https://www.html-code-generator.com/html/drop-down/country-region

    // Get the country name and state name from the imported script.
    let country_list = country_and_states['country'];
    let states_list = country_and_states['states'];

    // creating country name drop-down
    let option =  '';
    option += '<option>select country</option>';
    for(let country_code in country_list){
        // set selected option user country
        let selected = (country_code == user_country_code) ? ' selected' : '';
        option += '<option value="'+country_code+'"'+selected+'>'+country_list[country_code]+'</option>';
    }
    document.getElementById('country').innerHTML = option;

    // creating states name drop-down
    let text_box = '<input type="text" class="form-control" id="state">';
    let state_code_id = document.getElementById("state-code");

    function create_states_dropdown() {
        // get selected country code
        let country_code = document.getElementById("country").value;
        let states = states_list[country_code];
        // invalid country code or no states add textbox
        if(!states){
            state_code_id.innerHTML = text_box;
            return;
        }
        let option = '';
        if (states.length > 0) {
            option = '<select id="state" class="form-select">\n';
            for (let i = 0; i < states.length; i++) {
                if (country_code == 'NG' & states[i].code == 'NI'){
                    option += '<option selected="selected" value="'+states[i].code+'">'+states[i].name+'</option>';
                }
                else {
                    option += '<option value="'+states[i].code+'">'+states[i].name+'</option>';
                }
            }
            option += '</select>';
        } else {
            // create input textbox if no states 
            option = text_box
        }
        state_code_id.innerHTML = option;
    }

    // country select change event
    const country_select = document.getElementById("country");
    country_select.addEventListener('change', create_states_dropdown);

    create_states_dropdown();
})();