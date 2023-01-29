// put the settings into the web app
function import_settings_to_webapp(settings_dict) {
  document.getElementById("cost_pole").value = settings_dict.cost_pole;
  document.getElementById("cost_connection").value =
    settings_dict.cost_connection;
  document.getElementById("cost_distribution_cable").value =
    settings_dict.cost_distribution_cable;
  document.getElementById("cost_connection_cable").value =
    settings_dict.cost_connection_cable;
  document.getElementById("number_of_relaxation_steps_nr").value =
    settings_dict.number_of_relaxation_steps_nr;
}
