function resetForm() {
    const form = document.getElementById("predictForm");
    form.reset(); // clears form inputs

    // Also clear any pre-filled values rendered from Flask
    const inputs = form.querySelectorAll("input");
    inputs.forEach(input => input.value = "");

    // Remove prediction result section if present
    const result = document.querySelector(".result");
    if (result) result.remove();
}
