// ====================================================================
// SECTION: "Behind the Scenes" tab
// ====================================================================
// Everything on this tab is either static markup (bullets, the
// ensemble diagram) or a plain <img> pointing at a PNG the notebook
// already rendered -- the only backend call is this one GET /about,
// which just reads back a small precomputed JSON of dataset stats.
const aboutStatus = document.querySelector("#about-status");
const aboutStatsGrid = document.querySelector("#about-stats-grid");
const aboutCindex = document.querySelector("#about-cindex");

// Cached GET /about response; also doubles as the "already loaded"
// flag tabs.js checks before fetching again.
let aboutData = null;

// One value/label pair in the dataset-stats grid -- same small-card
// idea as buildFormulaColumn() in predict.js, just for plain stats
// instead of a percentile.
function buildStatItem(value, label) {
    const item = document.createElement("div");
    item.className = "stat-item";

    const valueEl = document.createElement("div");
    valueEl.className = "stat-value";
    valueEl.textContent = value;

    const labelEl = document.createElement("div");
    labelEl.className = "stat-label";
    labelEl.textContent = label;

    item.appendChild(valueEl);
    item.appendChild(labelEl);
    return item;
}

function renderAboutStats(data) {
    aboutStatsGrid.innerHTML = "";
    aboutStatsGrid.appendChild(buildStatItem(data.n_patients, "Patients"));
    aboutStatsGrid.appendChild(buildStatItem(
        data.median_visits + " (range " + data.min_visits + "–" + data.max_visits + ")",
        "Visits per patient"
    ));
    aboutStatsGrid.appendChild(buildStatItem(
        data.hepatic_event_count + " (" + data.hepatic_event_rate_pct + "%)",
        "Hepatic events"
    ));
    aboutStatsGrid.appendChild(buildStatItem(
        data.death_count + " (" + data.death_rate_pct + "%)",
        "Deaths"
    ));

    aboutCindex.textContent = data.weighted_cindex;
}

async function loadAboutStats() {
    aboutStatus.textContent = "Loading…";
    aboutStatus.className = "hint";

    try {
        const response = await fetch("/about");
        const data = await response.json();

        if (!response.ok) {
            aboutStatus.textContent = "Error: " + (data.detail || "Could not load stats.");
            aboutStatus.className = "hint status-error";
            return;
        }

        aboutData = data;
        renderAboutStats(data);
        aboutStatus.textContent = "";
    } catch (error) {
        aboutStatus.textContent = "Could not reach the server: " + error.message;
        aboutStatus.className = "hint status-error";
    }
}
