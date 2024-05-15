const mobile_nav = document.querySelector(".mobile-navbar-btn");
const nav_header = document.querySelector(".header");
var links=document.querySelectorAll(".header .navbar .navbar-list li")
var loader = document.getElementById('loading');
const toggleNavbar = () => {
  nav_header.classList.toggle("active");
}
const removeNavbar = () => {
    nav_header.classList.remove("active");
}

links.forEach((li) => {
  li.addEventListener("click", removeNavbar);
});
mobile_nav.addEventListener("click", () => toggleNavbar());
function loader_off(){
  setTimeout(function() {
      loader.style.display='none';
  }, 2000);
}

const toggleBtn = document.getElementById('toggleBtn');
const body = document.body;

toggleBtn.addEventListener('click', function() {
    body.classList.toggle('dark-mode');
});
