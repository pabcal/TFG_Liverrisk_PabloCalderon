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
        // never visits this tab. samplePatientsData is declared in
        // build-csv.js and starts out null, so this only fires once.
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

        // Same lazy-load idea for "Behind the Scenes": fetch GET /about
        // the first time this tab is opened. aboutData starts out null
        // (see about.js), so this only fires once.
        if (button.id === "tab-about" && !aboutData) {
            loadAboutStats();
        }
    });
});
