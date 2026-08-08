// ====================================================================
// SECTION: "Predict" tab -- DOM references
// ====================================================================
// Grab references to the page elements we need, once, at the top.
const uploadForm = document.querySelector("#upload-form");     // the <form> wrapping the file input + submit button
const fileInput = document.querySelector("#csv-file");         // the real (visually hidden) <input type="file">
const predictButton = document.querySelector("#predict-button"); // submit button, disabled while a request is in flight
const statusMessage = document.querySelector("#status-message"); // "Predicting...", "Done.", or an error line
const resultBox = document.querySelector("#result-box");       // <pre> that prints the raw JSON response
const fileNameLabel = document.querySelector("#file-name");    // fake "chosen file" text shown over the real input


// ====================================================================
// SECTION: "Predict" tab -- file input display
// ====================================================================
// Show the chosen file's name next to the (invisible) native input.
fileInput.addEventListener("change", function () {
    const chosenFile = fileInput.files[0];
    // Fall back to the placeholder text if the user cleared the picker
    fileNameLabel.textContent = chosenFile ? chosenFile.name : "No file chosen";
    // Toggles the "chosen" class, which just changes the label's color/weight
    fileNameLabel.classList.toggle("chosen", Boolean(chosenFile));
});


// ====================================================================
// SECTION: "Predict" tab -- form submit / POST /predict
// ====================================================================
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
    statusMessage.className = "";       // clear any leftover error/success styling
    resultBox.textContent = "";         // clear any previous result
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

    // Always re-enable the button, whether the request succeeded or failed.
    predictButton.disabled = false;
});


// ====================================================================
// SECTION: Tab switching (Predict <-> Build a test CSV)
// ====================================================================
// Both panels stay in the DOM the whole time; we just toggle the
// "hidden" class and the aria-selected state -- no page navigation,
// no framework router.
const tabButtons = document.querySelectorAll(".tab-button");

tabButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        // Mark only the clicked button as active/selected...
        tabButtons.forEach(function (b) {
            b.classList.toggle("active", b === button);
            b.setAttribute("aria-selected", b === button ? "true" : "false");
        });

        // ...and show only the panel that button controls (matched via
        // aria-labelledby, which we set to each button's id in the HTML).
        document.querySelectorAll(".tab-panel").forEach(function (panel) {
            const isTarget = panel.getAttribute("aria-labelledby") === button.id;
            panel.classList.toggle("hidden", !isTarget);
        });

        // Fetch the sample patients the first time this tab is opened,
        // not on page load -- no point calling the backend if the user
        // never visits this tab. samplePatientsData is declared further
        // down and starts out null, so this only fires once.
        if (button.id === "tab-build-csv" && !samplePatientsData) {
            loadSamplePatients();
        }
    });
});


// ====================================================================
// SECTION: "Build a test CSV" tab -- DOM references & state
// ====================================================================
const sampleStatus = document.querySelector("#sample-status");         // "Loading..." / error line above the cards
const patientList = document.querySelector("#patient-list");           // container the patient cards are rendered into
const csvPreview = document.querySelector("#csv-preview");             // read-only <textarea> showing the CSV so far
const csvPreviewHint = document.querySelector("#csv-preview-hint");    // "N patient(s) selected" line under the header
const downloadCsvButton = document.querySelector("#download-csv-button"); // disabled until >=1 patient is checked

// Holds the full /sample-patients response ({columns, patients}) once
// fetched, so every checkbox toggle can rebuild the CSV preview purely
// client-side, with no repeated backend calls.
let samplePatientsData = null;


// ====================================================================
// SECTION: "Build a test CSV" tab -- fetch sample patients
// ====================================================================
async function loadSamplePatients() {
    sampleStatus.textContent = "Loading sample patients...";
    sampleStatus.className = "";

    try {
        const response = await fetch("/sample-patients");
        const data = await response.json();

        if (!response.ok) {
            sampleStatus.textContent = "Error: " + data.detail;
            sampleStatus.className = "status-error";
            return;
        }

        // Cache the whole payload (column order + all 10 patients' raw
        // rows) so later checkbox toggles never need to fetch again.
        samplePatientsData = data;
        renderPatientCards(data.patients);
        sampleStatus.textContent = "";
    } catch (error) {
        // This happens if the server can't be reached at all.
        sampleStatus.textContent = "Could not reach the server: " + error.message;
        sampleStatus.className = "status-error";
    }
}


// ====================================================================
// SECTION: "Build a test CSV" tab -- render patient cards
// ====================================================================
function renderPatientCards(patients) {
    patientList.innerHTML = ""; // clear out any previous render

    patients.forEach(function (patient) {
        // Each card is a <label> wrapping the checkbox, so clicking
        // anywhere on the card (not just the tiny checkbox) toggles it,
        // keeping the tap target comfortably above 44px.
        const card = document.createElement("label");
        card.className = "patient-card";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "patient-checkbox";
        // Stash the patient's id on the checkbox itself so we can find
        // which patients are selected later without a separate lookup map.
        checkbox.dataset.trustiiId = String(patient.trustii_id);
        checkbox.addEventListener("change", updateCsvPreview);

        const info = document.createElement("div");
        info.className = "patient-info";

        const title = document.createElement("div");
        title.className = "patient-title";
        title.textContent = "Patient #" + patient.trustii_id;

        // Human-readable summary line(s), built from the fields the
        // backend precomputed (age, visit count, baseline FIB-4, most
        // complete NIT/lab) rather than from the raw per-visit columns.
        const summary = document.createElement("div");
        summary.className = "patient-summary";
        const fib4Text = patient.summary.baseline_fib4 === null
            ? "n/a"
            : patient.summary.baseline_fib4;
        summary.innerHTML =
            "Age at baseline: <strong>" + patient.summary.age_at_baseline + "</strong> &middot; " +
            "Visits recorded: <strong>" + patient.summary.visit_count + "</strong> &middot; " +
            "Baseline FIB-4: <strong>" + fib4Text + "</strong><br>" +
            "Most complete NIT/lab: <strong>" + patient.summary.most_complete_nit +
            "</strong> (" + patient.summary.most_complete_nit_visits + " visits)";

        info.appendChild(title);
        info.appendChild(summary);

        card.appendChild(checkbox);
        card.appendChild(info);
        patientList.appendChild(card);
    });
}


// ====================================================================
// SECTION: "Build a test CSV" tab -- CSV building
// ====================================================================
// Escapes one CSV field per RFC 4180: wrap in quotes (and double up any
// internal quotes) only when the value actually needs it.
function csvField(value) {
    if (value === null || value === undefined) {
        return ""; // missing values become an empty field, not the string "null"
    }
    const str = String(value);
    if (/[",\n]/.test(str)) {
        return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
}

// Builds the full CSV text (header row + one row per selected patient).
// `columns` is the exact column order from /sample-patients, and each
// row is built by looking up that same column name in patient.raw --
// so the header and the data can never drift out of alignment.
function buildCsvText(columns, selectedPatients) {
    const lines = [columns.map(csvField).join(",")];
    selectedPatients.forEach(function (patient) {
        const row = columns.map(function (col) {
            return csvField(patient.raw[col]);
        });
        lines.push(row.join(","));
    });
    return lines.join("\n");
}

// Rebuilds the CSV preview (and enables/disables the download button)
// from the checkboxes currently checked. Entirely client-side: uses
// the already-fetched samplePatientsData, no fetch() call here. Runs
// on every checkbox change event.
function updateCsvPreview() {
    // Collect the trustii_ids of every currently-checked card...
    const checkedBoxes = Array.from(document.querySelectorAll(".patient-checkbox:checked"));
    const checkedIds = new Set(checkedBoxes.map(function (cb) { return cb.dataset.trustiiId; }));
    // ...then filter the cached patient list down to just those.
    const selectedPatients = samplePatientsData.patients.filter(function (p) {
        return checkedIds.has(String(p.trustii_id));
    });

    if (selectedPatients.length === 0) {
        csvPreview.value = "";
        csvPreviewHint.textContent = "Select at least one patient above to build a file.";
        downloadCsvButton.disabled = true;
        return;
    }

    csvPreview.value = buildCsvText(samplePatientsData.columns, selectedPatients);
    csvPreviewHint.textContent = selectedPatients.length + " patient(s) selected.";
    downloadCsvButton.disabled = false;
}


// ====================================================================
// SECTION: "Build a test CSV" tab -- download button
// ====================================================================
downloadCsvButton.addEventListener("click", function () {
    // Build an in-memory file (Blob) from whatever text is currently
    // in the preview -- no need to recompute it, the preview always
    // mirrors the current selection.
    const blob = new Blob([csvPreview.value], { type: "text/csv" });
    const url = URL.createObjectURL(blob);

    // Trigger a download by clicking a throwaway <a download> link;
    // this is the standard way to save a Blob without a real backend
    // file to point at.
    const link = document.createElement("a");
    link.href = url;
    link.download = "liverrisk_sample_patients.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Release the temporary object URL now that the download has started.
    URL.revokeObjectURL(url);
});
