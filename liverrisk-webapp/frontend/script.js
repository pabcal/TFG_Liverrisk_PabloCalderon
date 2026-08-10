// ====================================================================
// SECTION: "Predict" tab -- DOM references
// ====================================================================
// Grab references to the page elements we need, once, at the top.
const uploadForm = document.querySelector("#upload-form");     // the <form> wrapping the file input + submit button
const fileInput = document.querySelector("#csv-file");         // the real (visually hidden) <input type="file">
const predictButton = document.querySelector("#predict-button"); // submit button, disabled while a request is in flight
const statusMessage = document.querySelector("#status-message"); // "Predicting...", "Done.", or an error line
const resultBox = document.querySelector("#result-box");       // container script.js fills with one .result-card per patient
const fileNameLabel = document.querySelector("#file-name");    // fake "chosen file" text shown over the real input

// "My active patients" pool for the Rankings tab: a plain in-memory
// array, not localStorage, so it resets on page refresh by design.
// Patients are added here from a result card's "+ Add to active pool"
// button (below) and read/ranked entirely client-side by the Rankings
// tab's own section further down -- nothing here ever touches the backend.
let activePatients = []; // each: {label, risk_hepatic_event, risk_death, weighted_risk, fib4_score, apri_score, age_at_baseline, visit_count}


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
    resultBox.innerHTML = "";           // clear any previous result cards
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
            renderResults(data);
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
// SECTION: "Predict" tab -- styled result card
// ====================================================================
// Turns a percentile number (0-100) into an ordinal string, e.g.
// 64.1 -> "64th", 21.7 -> "22nd". Rounds first, since "64.1th
// percentile" doesn't read as a normal English ordinal.
function ordinal(percentile) {
    const n = Math.round(percentile);
    const lastTwoDigits = n % 100;
    if (lastTwoDigits >= 11 && lastTwoDigits <= 13) {
        return n + "th";
    }
    switch (n % 10) {
        case 1: return n + "st";
        case 2: return n + "nd";
        case 3: return n + "rd";
        default: return n + "th";
    }
}

// Builds a plain-div bar histogram from the backend's pre-binned
// {bin_start, bin_end, count} array. Bar heights are set as a % of the
// tallest bin (inline style, computed here -- not a charting library).
// The one bar the patient's own score falls into is highlighted.
function buildHistogram(bins, value) {
    const container = document.createElement("div");
    container.className = "histogram";

    const maxCount = Math.max.apply(null, bins.map(function (bin) { return bin.count; }));

    bins.forEach(function (bin, i) {
        const bar = document.createElement("div");
        bar.className = "hist-bar";
        bar.style.height = (maxCount > 0 ? (bin.count / maxCount) * 100 : 0) + "%";

        // The last bin's upper edge is inclusive, so a patient scoring
        // exactly at the training cohort's max still lands in a bar.
        const isLastBin = i === bins.length - 1;
        const inThisBin = value !== null
            && value >= bin.bin_start
            && (isLastBin ? value <= bin.bin_end : value < bin.bin_end);
        if (inThisBin) {
            bar.classList.add("hist-bar-active");
        }

        container.appendChild(bar);
    });

    return container;
}

// One "Hepatic event risk" / "Death risk" sub-section: a percentile
// headline, the underlying blended score in small print, and a
// histogram marking where this patient falls in the training cohort.
// `note` (nullable) explains a flat-looking histogram, e.g. when that
// endpoint's blend is really just one model's rank-percentile alone --
// see main.py's single_model_note() -- so it doesn't read as broken.
function buildRiskSubsection(title, rawScore, percentile, histogramBins, note) {
    const section = document.createElement("div");
    section.className = "risk-subsection";

    const heading = document.createElement("h3");
    heading.textContent = title;

    const percentileLine = document.createElement("div");
    percentileLine.className = "risk-percentile";
    percentileLine.textContent = ordinal(percentile) + " percentile";

    const rawScoreLine = document.createElement("div");
    rawScoreLine.className = "risk-raw-score";
    rawScoreLine.textContent = "Blended risk score: " + rawScore;

    section.appendChild(heading);
    section.appendChild(percentileLine);
    section.appendChild(rawScoreLine);
    section.appendChild(buildHistogram(histogramBins, rawScore));

    if (note) {
        const noteLine = document.createElement("p");
        noteLine.className = "histogram-note";
        noteLine.textContent = note;
        section.appendChild(noteLine);
    }

    return section;
}

// One column ("ML blend" / "FIB-4" / "APRI") in the clinical-formula
// comparison. `percentile` is null when the patient is missing the
// labs a formula needs -- shown as plain text, not left blank.
function buildFormulaColumn(name, percentile) {
    const col = document.createElement("div");
    col.className = "formula-col";

    const label = document.createElement("div");
    label.className = "formula-name";
    label.textContent = name;

    const value = document.createElement("div");
    if (percentile === null) {
        value.className = "formula-value unavailable";
        value.textContent = "Not available (missing lab values)";
    } else {
        value.className = "formula-value";
        value.textContent = ordinal(percentile) + " percentile";
    }

    col.appendChild(label);
    col.appendChild(value);
    return col;
}

// Row that pushes this patient's already-computed scores into the
// in-memory activePatients pool (see the DOM-references section at the
// top), for the Rankings tab's "My active patients" scope. Uses a
// plain inline text input for the label, defaulting to "Patient N" if
// left blank -- NOT window.prompt()/confirm(), which some embedded
// browser contexts (e.g. VS Code's Simple Browser) silently block,
// making a prompt()-based button look like it does nothing at all.
function buildAddToPoolRow(result) {
    const row = document.createElement("div");
    row.className = "add-to-pool-row";

    const defaultLabel = "Patient " + (activePatients.length + 1);

    const labelInput = document.createElement("input");
    labelInput.type = "text";
    labelInput.className = "add-to-pool-input";
    labelInput.placeholder = defaultLabel;
    labelInput.setAttribute("aria-label", "Label for this patient in your active pool");

    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "add-to-pool-button";
    addButton.textContent = "+ Add to active pool";

    addButton.addEventListener("click", function () {
        const enteredLabel = labelInput.value.trim();

        activePatients.push({
            label: enteredLabel === "" ? defaultLabel : enteredLabel,
            risk_hepatic_event: result.risk_hepatic_event,
            risk_death: result.risk_death,
            weighted_risk: result.weighted_risk,
            fib4_score: result.fib4_score,
            apri_score: result.apri_score,
            age_at_baseline: result.age_at_baseline,
            visit_count: result.visit_count,
        });

        addButton.textContent = "Added ✓";
        addButton.disabled = true;
        labelInput.disabled = true;
    });

    row.appendChild(labelInput);
    row.appendChild(addButton);
    return row;
}

// Builds one full .result-card for one patient's /predict result.
// `index`/`total` are only used to label the card "Patient N of M"
// when the uploaded CSV had more than one patient.
function buildResultCard(result, index, total) {
    const card = document.createElement("div");
    card.className = "result-card";

    if (total > 1) {
        const cardTitle = document.createElement("div");
        cardTitle.className = "result-card-title";
        cardTitle.textContent = "Patient " + (index + 1) + " of " + total;
        card.appendChild(cardTitle);
    }

    // The single headline figure: overall (weighted) risk percentile.
    const weighted = document.createElement("div");
    weighted.className = "weighted-risk";

    const weightedNumber = document.createElement("div");
    weightedNumber.className = "weighted-risk-number";
    weightedNumber.textContent = ordinal(result.weighted_percentile);

    const weightedLabel = document.createElement("div");
    weightedLabel.className = "weighted-risk-label";
    weightedLabel.textContent =
        "This patient ranks in the " + ordinal(result.weighted_percentile) +
        " percentile for overall risk (a 70% hepatic-event / 30% death blend), compared to the training cohort.";

    weighted.appendChild(weightedNumber);
    weighted.appendChild(weightedLabel);
    card.appendChild(weighted);

    // Hepatic event / death risk, side by side, each with its own histogram.
    const subsections = document.createElement("div");
    subsections.className = "risk-subsections";
    subsections.appendChild(buildRiskSubsection(
        "Hepatic event risk", result.risk_hepatic_event, result.hepatic_percentile,
        result.histograms.hepatic, result.distribution_notes.hepatic
    ));
    subsections.appendChild(buildRiskSubsection(
        "Death risk", result.risk_death, result.death_percentile,
        result.histograms.death, result.distribution_notes.death
    ));
    card.appendChild(subsections);

    // Hepatic-risk percentile from the ML blend next to the two
    // formula-based scores, so they can be read side by side.
    const comparison = document.createElement("div");
    comparison.className = "formula-comparison";

    const comparisonHeading = document.createElement("h3");
    comparisonHeading.textContent = "Compared to clinical formulas";
    comparison.appendChild(comparisonHeading);

    const columns = document.createElement("div");
    columns.className = "formula-columns";
    columns.appendChild(buildFormulaColumn("ML blend (hepatic)", result.hepatic_percentile));
    columns.appendChild(buildFormulaColumn("FIB-4", result.fib4_percentile));
    columns.appendChild(buildFormulaColumn("APRI", result.apri_percentile));
    comparison.appendChild(columns);
    card.appendChild(comparison);

    card.appendChild(buildAddToPoolRow(result));

    return card;
}

// Renders the full /predict response (a list of per-patient results)
// as one result card per patient, replacing whatever was in the box.
function renderResults(results) {
    resultBox.innerHTML = "";
    results.forEach(function (result, i) {
        resultBox.appendChild(buildResultCard(result, i, results.length));
    });
}


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

        // Same lazy-load idea for Rankings: fetch/render the first time
        // the tab is opened, not at page load. Safe to call again on
        // every later visit too -- renderRankingsTable() re-renders from
        // its own cache (training rows) or the in-memory active-patients
        // array, with no repeat backend call either way.
        if (button.id === "tab-rankings") {
            renderRankingsTable();
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


// ====================================================================
// SECTION: "Rankings" tab -- DOM references & state
// ====================================================================
const rankingsStatus = document.querySelector("#rankings-status");
const rankingsThead = document.querySelector("#rankings-thead");
const rankingsTbody = document.querySelector("#rankings-tbody");
const hideMissingRow = document.querySelector("#hide-missing-row");
const hideMissingCheckbox = document.querySelector("#hide-missing-checkbox");
const methodButtons = document.querySelectorAll(".toggle-button[data-method]");
const scopeButtons = document.querySelectorAll(".toggle-button[data-scope]");

// Defaults match the HTML's "active" toggle buttons: ML blend, training cohort.
let rankingsMethod = "ml";   // "ml" | "fib4" | "apri"
let rankingsScope = "training"; // "training" | "active"

// GET /rankings responses, cached per method so switching back and
// forth between ML/FIB-4/APRI doesn't re-fetch data that never changes
// while the server is running. activePatients (the other data source)
// is declared up in the Predict-tab section, since it's filled in from there.
let rankingsTrainingCache = {}; // method -> array of row objects


// ====================================================================
// SECTION: "Rankings" tab -- shared helpers
// ====================================================================
// Human-readable label for the currently selected method, used in the
// table's "Score" column header.
function methodLabel(method) {
    if (method === "fib4") return "FIB-4";
    if (method === "apri") return "APRI";
    return "ML blend";
}

// Which of an active-pool patient's already-computed scores to rank by,
// matching whichever method is currently toggled -- the same method
// toggle drives both scopes, just from a different data source.
function activePatientScore(patient, method) {
    if (method === "fib4") return patient.fib4_score;
    if (method === "apri") return patient.apri_score;
    return patient.weighted_risk;
}

// A plain <td> with the given text.
function buildCell(text) {
    const cell = document.createElement("td");
    cell.textContent = text;
    return cell;
}


// ====================================================================
// SECTION: "Rankings" tab -- fetch training-cohort rankings
// ====================================================================
async function loadTrainingRanking(method) {
    if (rankingsTrainingCache[method]) {
        return rankingsTrainingCache[method];
    }

    const response = await fetch("/rankings?method=" + method + "&scope=training");
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || "Unknown error.");
    }

    rankingsTrainingCache[method] = data.rows;
    return data.rows;
}


// ====================================================================
// SECTION: "Rankings" tab -- render the unified table
// ====================================================================
// Header row for the "Training cohort" scope.
function renderTrainingTableHead() {
    const tr = document.createElement("tr");
    ["Rank", "Patient ID", "Score (" + methodLabel(rankingsMethod) + ")", "Age", "Visits", "Outcome"].forEach(function (label) {
        const th = document.createElement("th");
        th.textContent = label;
        tr.appendChild(th);
    });
    rankingsThead.appendChild(tr);
}

// Header row for the "My active patients" scope (no percentile column,
// a blank header cell over the remove buttons instead of "Outcome").
function renderActiveTableHead() {
    const tr = document.createElement("tr");
    ["Rank", "Patient", "Score (" + methodLabel(rankingsMethod) + ")", "Age", "Visits", ""].forEach(function (label) {
        const th = document.createElement("th");
        th.textContent = label;
        tr.appendChild(th);
    });
    rankingsThead.appendChild(tr);
}

// "0.71 (71st percentile)", or a plain note when the score is missing
// (a patient without the labs FIB-4/APRI need).
function formatScoreWithPercentile(score, percentile) {
    if (score === null) {
        return "Missing lab values";
    }
    return score.toFixed(2) + " (" + ordinal(percentile) + " percentile)";
}

// One <td> for the training-cohort "Outcome" column: a small muted tag
// for a real event, or a plain "No event recorded" note otherwise.
function buildOutcomeCell(outcome) {
    const cell = document.createElement("td");
    if (outcome) {
        const tag = document.createElement("span");
        tag.className = "outcome-tag";
        tag.textContent = outcome;
        cell.appendChild(tag);
    } else {
        const span = document.createElement("span");
        span.className = "no-outcome";
        span.textContent = "No event recorded";
        cell.appendChild(span);
    }
    return cell;
}

// Renders the "Training cohort" scope: fetches (or reuses the cached)
// /rankings rows for the current method, optionally hides null-score
// rows, and fills in the table.
async function renderTrainingScope() {
    rankingsStatus.textContent = "Loading rankings...";
    rankingsStatus.className = "hint";

    let rows;
    try {
        rows = await loadTrainingRanking(rankingsMethod);
    } catch (error) {
        rankingsStatus.textContent = "Could not load rankings: " + error.message;
        rankingsStatus.className = "hint status-error";
        return;
    }
    rankingsStatus.textContent = "";

    if (hideMissingCheckbox.checked && rankingsMethod !== "ml") {
        rows = rows.filter(function (row) { return row.score !== null; });
    }

    renderTrainingTableHead();
    rows.forEach(function (row) {
        const tr = document.createElement("tr");
        tr.appendChild(buildCell(String(row.rank)));
        tr.appendChild(buildCell(row.patient_id));
        tr.appendChild(buildCell(formatScoreWithPercentile(row.score, row.percentile)));
        tr.appendChild(buildCell(row.age_at_baseline === null ? "n/a" : String(row.age_at_baseline)));
        tr.appendChild(buildCell(String(row.visit_count)));
        tr.appendChild(buildOutcomeCell(row.outcome));
        rankingsTbody.appendChild(tr);
    });
}

// Renders the "My active patients" scope: ranks the in-memory
// activePatients array client-side by whichever method is selected --
// no fetch at all.
function renderActiveScope() {
    if (activePatients.length === 0) {
        rankingsStatus.textContent = "No active patients yet — go to Predict, score a patient, and click \"Add to active pool.\"";
        rankingsStatus.className = "hint";
        renderActiveTableHead();
        return;
    }
    rankingsStatus.textContent = "";

    // Pair each patient with its score for the current method, keeping
    // the original array index so a Remove click can splice() the right entry.
    let entries = activePatients.map(function (patient, index) {
        return { patient: patient, index: index, score: activePatientScore(patient, rankingsMethod) };
    });

    if (hideMissingCheckbox.checked && rankingsMethod !== "ml") {
        entries = entries.filter(function (entry) { return entry.score !== null; });
    }

    // Descending by score, null-score entries always last.
    entries.sort(function (a, b) {
        if ((a.score === null) !== (b.score === null)) {
            return a.score === null ? 1 : -1;
        }
        return a.score === null ? 0 : b.score - a.score;
    });

    renderActiveTableHead();
    entries.forEach(function (entry, position) {
        const tr = document.createElement("tr");
        tr.appendChild(buildCell(String(position + 1)));
        tr.appendChild(buildCell(entry.patient.label));
        tr.appendChild(buildCell(entry.score === null ? "Missing lab values" : entry.score.toFixed(2)));
        tr.appendChild(buildCell(entry.patient.age_at_baseline === null ? "n/a" : String(entry.patient.age_at_baseline)));
        tr.appendChild(buildCell(String(entry.patient.visit_count)));

        const removeCell = document.createElement("td");
        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.className = "remove-button";
        removeButton.textContent = "×";
        removeButton.setAttribute("aria-label", "Remove " + entry.patient.label + " from active pool");
        removeButton.addEventListener("click", function () {
            // entry.index is this patient's position in the *original*
            // activePatients array (captured before this render's sort/filter).
            activePatients.splice(entry.index, 1);
            renderRankingsTable();
        });
        removeCell.appendChild(removeButton);
        tr.appendChild(removeCell);

        rankingsTbody.appendChild(tr);
    });
}

// Top-level render function for the whole tab: clears the table, shows
// the hide-missing checkbox only when it's actually meaningful (FIB-4/
// APRI, where a score can be null), then delegates to whichever scope
// is currently selected.
async function renderRankingsTable() {
    rankingsThead.innerHTML = "";
    rankingsTbody.innerHTML = "";
    hideMissingRow.classList.toggle("hidden", rankingsMethod === "ml");

    if (rankingsScope === "training") {
        await renderTrainingScope();
    } else {
        renderActiveScope();
    }
}


// ====================================================================
// SECTION: "Rankings" tab -- toggle buttons
// ====================================================================
methodButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        methodButtons.forEach(function (b) { b.classList.toggle("active", b === button); });
        rankingsMethod = button.dataset.method;
        renderRankingsTable();
    });
});

scopeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        scopeButtons.forEach(function (b) { b.classList.toggle("active", b === button); });
        rankingsScope = button.dataset.scope;
        renderRankingsTable();
    });
});

hideMissingCheckbox.addEventListener("change", renderRankingsTable);
