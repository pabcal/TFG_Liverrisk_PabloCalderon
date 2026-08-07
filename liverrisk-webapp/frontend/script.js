// Grab references to the page elements we need, once, at the top.
const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#csv-file");
const predictButton = document.querySelector("#predict-button");
const statusMessage = document.querySelector("#status-message");
const resultBox = document.querySelector("#result-box");
const fileNameLabel = document.querySelector("#file-name");

// Show the chosen file's name next to the (invisible) native input.
fileInput.addEventListener("change", function () {
    const chosenFile = fileInput.files[0];
    fileNameLabel.textContent = chosenFile ? chosenFile.name : "No file chosen";
    fileNameLabel.classList.toggle("chosen", Boolean(chosenFile));
});

// Runs every time the user submits the form (clicks "Predict").
uploadForm.addEventListener("submit", async function (event) {
    // Stop the browser's default behavior, which would reload the page.
    event.preventDefault();

    // Make sure a file was actually chosen.
    const chosenFile = fileInput.files[0];
    if (!chosenFile) {
        statusMessage.textContent = "Please choose a CSV file first.";
        return;
    }

    // Show a loading message and disable the button so the user can't
    // click it twice while we're waiting for the server.
    statusMessage.textContent = "Predicting...";
    statusMessage.className = "";
    resultBox.textContent = "";
    predictButton.disabled = true;

    // FormData is the standard way to send a file to a server with fetch().
    const formData = new FormData();
    formData.append("file", chosenFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        // The backend sends a JSON body either way (result on success,
        // {"detail": "..."} on error), so we can parse it either way.
        const data = await response.json();

        if (!response.ok) {
            // FastAPI puts error messages in a "detail" field.
            statusMessage.textContent = "Error: " + data.detail;
            statusMessage.className = "status-error";
        } else {
            statusMessage.textContent = "Done.";
            statusMessage.className = "status-success";
            // Pretty-print the JSON with 2-space indentation so it's readable.
            resultBox.textContent = JSON.stringify(data, null, 2);
        }
    } catch (error) {
        // This happens if the server can't be reached at all.
        statusMessage.textContent = "Could not reach the server: " + error.message;
        statusMessage.className = "status-error";
    }

    predictButton.disabled = false;
});
