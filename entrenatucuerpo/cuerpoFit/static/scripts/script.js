document.addEventListener("DOMContentLoaded", function() {
  const value = document.querySelector("#vSpinner");
  const input = document.querySelector("#Heart_Rate");
  value.textContent = input.value;

  input.addEventListener("input", (event) => {
    value.textContent = event.target.value;
  });
});

