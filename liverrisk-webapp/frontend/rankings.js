// ====================================================================
// SECTION: "Rankings" tab -- DOM references & state
// ====================================================================
const rankingsStatus = document.querySelector("#rankings-status");
const rankingsThead = document.querySelector("#rankings-thead");
const rankingsTbody = document.querySelector("#rankings-tbody");
const hideMissingRow = document.querySelector("#hide-missing-row");
const hideMissingCheckbox = document.querySelector("#hide-missing-checkbox");
const methodToggleGroup = document.querySelector("#method-toggle-group");
const methodButtons = document.querySelectorAll(".toggle-button[data-method]");
const scopeButtons = document.querySelectorAll(".toggle-button[data-scope]");
const disagreementFormulaGroup = document.querySelector("#disagreement-formula-group");
const formulaButtons = document.querySelectorAll(".toggle-button[data-formula]");
const disagreementCaption = document.querySelector("#disagreement-caption");

// Defaults match the HTML's "active" toggle buttons: ML blend, training
// cohort, FIB-4 (the latter only actually used once scope=disagreement).
let rankingsMethod = "ml";   // "ml" | "fib4" | "apri"
let rankingsScope = "training"; // "training" | "active" | "disagreement"
let rankingsFormula = "fib4"; // "fib4" | "apri" -- only meaningful for scope=disagreement

// GET /rankings responses, cached per method so switching back and
// forth between ML/FIB-4/APRI doesn't re-fetch data that never changes
// while the server is running. activePatients (the other data source)
// is declared in state.js, since it's filled in from the Predict tab.
let rankingsTrainingCache = {}; // method -> array of row objects

// GET /disagreement responses, cached per formula the same way.
let disagreementTrainingCache = {}; // formula -> array of row objects


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
// SECTION: "Rankings" tab -- fetch disagreement rankings
// ====================================================================
// Static training-cohort disagreement rows (no new patient), cached
// per formula just like loadTrainingRanking() above.
async function loadDisagreementTraining(formula) {
    if (disagreementTrainingCache[formula]) {
        return disagreementTrainingCache[formula];
    }

    const response = await fetch("/disagreement?formula=" + formula);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || "Unknown error.");
    }

    disagreementTrainingCache[formula] = data.rows;
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

// Header row for the "Disagreement" scope.
function renderDisagreementTableHead() {
    const tr = document.createElement("tr");
    ["Patient ID", "ML Rank", methodLabel(rankingsFormula) + " Rank", "Rank Gap", "Event", "Time"].forEach(function (label) {
        const th = document.createElement("th");
        th.textContent = label;
        tr.appendChild(th);
    });
    rankingsThead.appendChild(tr);
}

// "+42" / "-15" -- explicit sign so a positive gap always reads as
// positive, not just as a plain number.
function formatRankGap(rankGap) {
    const rounded = Math.round(rankGap);
    return (rounded > 0 ? "+" : "") + String(rounded);
}

// Renders the "Disagreement" scope: fetches (or reuses the cached)
// /disagreement rows for the current formula and fills in the table.
async function renderDisagreementScope() {
    rankingsStatus.textContent = "Loading rankings...";
    rankingsStatus.className = "hint";

    let rows;
    try {
        rows = await loadDisagreementTraining(rankingsFormula);
    } catch (error) {
        rankingsStatus.textContent = "Could not load rankings: " + error.message;
        rankingsStatus.className = "hint status-error";
        return;
    }
    rankingsStatus.textContent = "";

    renderDisagreementTableHead();
    rows.forEach(function (row) {
        const tr = document.createElement("tr");
        tr.appendChild(buildCell(row.patient_id));
        tr.appendChild(buildCell(String(row.ml_rank)));
        tr.appendChild(buildCell(String(row.formula_rank)));
        tr.appendChild(buildCell(formatRankGap(row.rank_gap)));
        tr.appendChild(buildCell(row.event === null ? "—" : (row.event ? "Event" : "No event")));
        tr.appendChild(buildCell(row.time === null ? "—" : row.time.toFixed(1) + " yr"));
        rankingsTbody.appendChild(tr);
    });
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
// APRI training/active scopes, where a score can be null), swaps the
// method toggle for the formula toggle (and shows the sign-convention
// caption) only for the Disagreement scope, then delegates to whichever
// scope is currently selected.
async function renderRankingsTable() {
    rankingsThead.innerHTML = "";
    rankingsTbody.innerHTML = "";
    hideMissingRow.classList.toggle("hidden", rankingsMethod === "ml" || rankingsScope === "disagreement");

    const isDisagreement = rankingsScope === "disagreement";
    methodToggleGroup.classList.toggle("hidden", isDisagreement);
    disagreementFormulaGroup.classList.toggle("hidden", !isDisagreement);
    disagreementCaption.classList.toggle("hidden", !isDisagreement);
    if (isDisagreement) {
        disagreementCaption.textContent =
            "Positive gap = ML flags this patient as more urgent than " + methodLabel(rankingsFormula) + "; negative = the reverse.";
    }

    if (rankingsScope === "training") {
        await renderTrainingScope();
    } else if (rankingsScope === "active") {
        renderActiveScope();
    } else {
        await renderDisagreementScope();
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

// Sets the scope state and its toggle button's active state, without
// re-rendering -- shared by the click listener below and by predict.js,
// which activates this scope programmatically from the Predict tab's
// "See where this patient ranks" button before switching to this tab.
function activateScope(scope) {
    scopeButtons.forEach(function (b) { b.classList.toggle("active", b.dataset.scope === scope); });
    rankingsScope = scope;
}

scopeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        activateScope(button.dataset.scope);
        renderRankingsTable();
    });
});

formulaButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        formulaButtons.forEach(function (b) { b.classList.toggle("active", b === button); });
        rankingsFormula = button.dataset.formula;
        renderRankingsTable();
    });
});

hideMissingCheckbox.addEventListener("change", renderRankingsTable);
