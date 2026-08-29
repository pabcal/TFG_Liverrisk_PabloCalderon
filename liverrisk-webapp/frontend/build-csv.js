// ====================================================================
// SECTION: "Build a test CSV" tab -- DOM references & state
// ====================================================================
const sampleStatus = document.querySelector("#sample-status");         // "Loading..." / error line above the cards
const patientList = document.querySelector("#patient-list");           // container the patient cards are rendered into
const csvPreviewThead = document.querySelector("#csv-preview-thead");  // <thead> of the read-only table preview
const csvPreviewTbody = document.querySelector("#csv-preview-tbody");  // <tbody> of the read-only table preview
const csvPreviewHint = document.querySelector("#csv-preview-hint");    // "N patient(s) selected" line under the header
const downloadCsvButton = document.querySelector("#download-csv-button"); // disabled until >=1 patient is checked

// Holds the full /sample-patients response ({columns, patients}) once
// fetched, so every checkbox toggle can rebuild the CSV preview purely
// client-side, with no repeated backend calls.
let samplePatientsData = null;

// Holds the exact CSV text for the current selection (built alongside
// the table preview), so the download button always saves the real,
// unmodified CSV rather than anything derived from the displayed table.
let currentCsvText = "";


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

// Renders the same header/row data as buildCsvText(), but as a real
// <table> for display: one <th> per column, missing values shown as
// "--" instead of an empty stretch between commas.
function renderCsvPreviewTable(columns, selectedPatients) {
    csvPreviewThead.innerHTML = "";
    csvPreviewTbody.innerHTML = "";

    const headerRow = document.createElement("tr");
    columns.forEach(function (col) {
        const th = document.createElement("th");
        th.textContent = col;
        headerRow.appendChild(th);
    });
    csvPreviewThead.appendChild(headerRow);

    selectedPatients.forEach(function (patient) {
        const row = document.createElement("tr");
        columns.forEach(function (col) {
            const td = document.createElement("td");
            const value = patient.raw[col];
            const isMissing = value === null || value === undefined || value === "";
            td.textContent = isMissing ? "--" : String(value);
            if (isMissing) {
                td.className = "missing-value";
            }
            row.appendChild(td);
        });
        csvPreviewTbody.appendChild(row);
    });
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
        currentCsvText = "";
        csvPreviewThead.innerHTML = "";
        csvPreviewTbody.innerHTML = "";
        csvPreviewHint.textContent = "Select at least one patient above to build a file.";
        downloadCsvButton.disabled = true;
        return;
    }

    currentCsvText = buildCsvText(samplePatientsData.columns, selectedPatients);
    renderCsvPreviewTable(samplePatientsData.columns, selectedPatients);
    csvPreviewHint.textContent = selectedPatients.length + " patient(s) selected.";
    downloadCsvButton.disabled = false;
}


// ====================================================================
// SECTION: "Build a test CSV" tab -- download button
// ====================================================================
downloadCsvButton.addEventListener("click", function () {
    // Build an in-memory file (Blob) from the current CSV text -- no
    // need to recompute it, currentCsvText always mirrors the current
    // selection (kept in sync with the table preview by updateCsvPreview).
    const blob = new Blob([currentCsvText], { type: "text/csv" });
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
