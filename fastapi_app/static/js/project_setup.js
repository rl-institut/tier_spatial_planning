var currentDate = new Date();
var Year = currentDate.getFullYear();
var firstDay = new Date(Date.UTC(Year, 0, 1));
var startDateInput = document.getElementById("startDate");
startDateInput.value = firstDay.toISOString().slice(0,10);