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

// What fraction of the training cohort an inserted active-pool patient
// ranks above, computed exactly the way the backend's percentile_le()
// computes every training row's percentile (models_loader.py): the
// percentage of reference scores <= this score, rounded to 1 decimal.
function percentileAmongTraining(score, trainingRows) {
    const scored = trainingRows.filter(function (row) { return row.score !== null; });
    if (scored.length === 0) {
        return null;
    }
    const countLE = scored.filter(function (row) { return row.score <= score; }).length;
    return Math.round((countLE / scored.length) * 1000) / 10;
}

// Builds the full merged list -- every training row plus one row per
// activePatients entry, inserted wherever its score sorts -- shared by
// the "Full transplant list" table (renderTrainingScope) and the "My
// active patients" scope's "Rank in full list" column (renderActiveScope).
// Never mutates trainingRows or recomputes any training row's own
// score/percentile. A row's displayed rank must always come from its
// *position in this merged, sorted array*, never from a training row's
// original backend-assigned `rank` field -- once patients are inserted,
// that original field is stale for every row at or below the insertion
// point (two rows can otherwise show the same number).
function buildMergedRankedRows(trainingRows, method) {
    let rows = trainingRows.slice();
    activePatients.forEach(function (patient) {
        const score = activePatientScore(patient, method);
        rows.push({
            patient_id: patient.label,
            score: score,
            percentile: score === null ? null : percentileAmongTraining(score, trainingRows),
            age_at_baseline: patient.age_at_baseline,
            visit_count: patient.visit_count,
            outcome: null,
            isInserted: true,
            patient: patient,
        });
    });

    // Descending by score, null-score rows always last -- same
    // convention build_training_ranking_rows() uses server-side.
    rows.sort(function (a, b) {
        if ((a.score === null) !== (b.score === null)) {
            return a.score === null ? 1 : -1;
        }
        return a.score === null ? 0 : b.score - a.score;
    });

    return rows;
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
// "Rank in full list" is this patient's 1-based position in the full
// merged (training + active) list -- see buildMergedRankedRows().
function renderActiveTableHead() {
    const tr = document.createElement("tr");
    ["Rank", "Patient", "Score (" + methodLabel(rankingsMethod) + ")", "Rank in full list", "Age", "Visits", ""].forEach(function (label) {
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

// Score cell for one row in the "Full transplant list" table. Training
// rows carry their backend-computed percentile; inserted active-pool
// rows get theirs from percentileAmongTraining() in
// buildMergedRankedRows(), computed the same way -- so every row goes
// through the same formatting helper.
function formatScoreCell(row) {
    return formatScoreWithPercentile(row.score, row.percentile);
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

// Renders the "Full transplant list" scope (internally still scope
// "training" -- see activateScope/index.html, only the button's label
// changed): fetches (or reuses the cached) /rankings rows for the
// current method, merges in every activePatients entry at its computed
// insertion rank, optionally hides null-score rows, and fills in the
// table. The merge is entirely client-side -- no fetch beyond the one
// loadTrainingRanking() call, and no recomputation of any training
// row's own rank/score/percentile.
async function renderTrainingScope() {
    rankingsStatus.textContent = "Loading rankings...";
    rankingsStatus.className = "hint";

    let trainingRows;
    try {
        trainingRows = await loadTrainingRanking(rankingsMethod);
    } catch (error) {
        rankingsStatus.textContent = "Could not load rankings: " + error.message;
        rankingsStatus.className = "hint status-error";
        return;
    }
    rankingsStatus.textContent = "";

    // Merge in every active-pool patient at its sorted position (never
    // mutates the cached trainingRows array).
    let rows = buildMergedRankedRows(trainingRows, rankingsMethod);

    if (hideMissingCheckbox.checked && rankingsMethod !== "ml") {
        rows = rows.filter(function (row) { return row.score !== null; });
    }

    renderTrainingTableHead();
    // Rank displayed is always this row's 1-based position in the final
    // rendered array -- NOT row.rank / any training row's original
    // backend-assigned rank, which is stale for every row at or below
    // wherever a patient got inserted.
    rows.forEach(function (row, position) {
        const tr = document.createElement("tr");
        if (row.isInserted) {
            tr.classList.add("inserted-patient");
        }
        tr.appendChild(buildCell(String(position + 1)));
        tr.appendChild(buildCell(row.patient_id));
        tr.appendChild(buildCell(formatScoreCell(row)));
        tr.appendChild(buildCell(row.age_at_baseline === null ? "n/a" : String(row.age_at_baseline)));
        tr.appendChild(buildCell(String(row.visit_count)));
        tr.appendChild(buildOutcomeCell(row.outcome));
        rankingsTbody.appendChild(tr);
    });
}

// Renders the "My active patients" scope: ranks the in-memory
// activePatients array client-side by whichever method is selected, and
// shows each patient's "Rank in full list" (buildMergedRankedRows()
// against the training cohort -- the same helper renderTrainingScope
// uses). That column needs the current method's training rows, which
// this scope otherwise never fetches -- loadTrainingRanking() is a
// no-op past the first call (cached in rankingsTrainingCache), so this
// only ever hits the network once per method, same as every other
// caller.
async function renderActiveScope() {
    if (activePatients.length === 0) {
        rankingsStatus.textContent = "No active patients yet — go to Predict, score a patient, and click \"Add to active pool.\"";
        rankingsStatus.className = "hint";
        renderActiveTableHead();
        return;
    }
    rankingsStatus.textContent = "";

    let trainingRows = null;
    try {
        trainingRows = await loadTrainingRanking(rankingsMethod);
    } catch (error) {
        // "Rank in full list" just shows "—" below when trainingRows is
        // null -- not fatal to the rest of this scope, which needs no
        // backend data at all.
    }

    // Build the same merged, sorted full list renderTrainingScope shows,
    // so each patient's "Rank in full list" is its 1-based position in
    // that array -- accounting for every other active patient that also
    // ranks above it, not just training rows.
    const mergedRows = trainingRows ? buildMergedRankedRows(trainingRows, rankingsMethod) : null;

    // Pair each patient with its score for the current method, keeping
    // the original array index so a Remove click can splice() the right entry.
    let entries = activePatients.map(function (patient, index) {
        const score = activePatientScore(patient, rankingsMethod);
        const fullListRank = mergedRows ? mergedRows.findIndex(function (row) { return row.patient === patient; }) + 1 : 0;
        return {
            patient: patient,
            index: index,
            score: score,
            fullListRank: fullListRank > 0 ? fullListRank : null,
        };
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
        tr.appendChild(buildCell(entry.fullListRank === null ? "—" : String(entry.fullListRank)));
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
    //if (isDisagreement) {
    //    disagreementCaption.textContent =
    //        "Positive gap = ML flags this patient as more urgent than " + methodLabel(rankingsFormula) + "; negative = the reverse.";
    //}

    if (rankingsScope === "training") {
        await renderTrainingScope();
    } else if (rankingsScope === "active") {
        await renderActiveScope();
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
