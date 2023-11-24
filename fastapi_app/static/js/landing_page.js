document.getElementById('myForm').addEventListener('submit', function(e) {
  e.preventDefault();
  document.getElementById('myButton').click();
});
$(window).scroll(function() {
    var scrollTop = $(this).scrollTop();
    $('.parallax').css('transform', 'translateY(' + -(scrollTop * 0.2) + 'px)');
});

  ScrollReveal().reveal('.col-md-6:nth-child(odd):not(.no-reveal)', {
    origin: 'left',
    distance: '0px',
    duration: 1000,
    reset: false
  });

  ScrollReveal().reveal('.col-md-6:nth-child(even):not(.no-reveal)', {
    origin: 'right',
    distance: '0px',
    duration: 1000,
    reset: false
  });