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
// see models_loader.py's single_model_note() -- so it doesn't read as broken.
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

// "Why this score" section: a plain-language sentence naming this
// patient's top SHAP-ranked features (result.risk_explanation, computed
// live per patient in explain.py). Falls back to a plain unavailable
// line instead of an empty section when the backend couldn't compute one.
function buildRiskExplanation(result) {
    const section = document.createElement("div");
    section.className = "risk-explanation";

    const heading = document.createElement("h3");
    heading.textContent = "Why this score";
    section.appendChild(heading);

    const text = document.createElement("p");
    if (result.risk_explanation) {
        text.className = "risk-explanation-text";
        text.textContent = result.risk_explanation;
    } else {
        text.className = "risk-explanation-text unavailable";
        text.textContent = "Explanation not available for this patient.";
    }
    section.appendChild(text);

    return section;
}

// Row that pushes this patient's already-computed scores into the
// in-memory activePatients pool (see state.js), for the Rankings tab's
// "My active patients" scope. Uses a plain inline text input for the
// label, defaulting to "Patient N" if left blank -- NOT
// window.prompt()/confirm(), which some embedded browser contexts
// (e.g. VS Code's Simple Browser) silently block, making a
// prompt()-based button look like it does nothing at all.
function buildAddToPoolRow(result) {
    const row = document.createElement("div");
    row.className = "add-to-pool-row";

    const labelInput = document.createElement("input");
    labelInput.type = "text";
    labelInput.className = "add-to-pool-input";
    labelInput.placeholder = "Patient " + (activePatients.length + 1);
    labelInput.setAttribute("aria-label", "Label for this patient in your active pool");

    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "add-to-pool-button";
    addButton.textContent = "+ Add to active pool";

    addButton.addEventListener("click", function () {
        const enteredLabel = labelInput.value.trim();
        // Computed at click time, not build time -- several cards can sit
        // unadded together (a multi-patient CSV), so activePatients.length
        // at render time would give every one of them the same "Patient 1"
        // fallback instead of counting up as they're actually added.
        const defaultLabel = "Patient " + (activePatients.length + 1);

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

    card.appendChild(buildRiskExplanation(result));

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
