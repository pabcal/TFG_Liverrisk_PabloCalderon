// ====================================================================
// SECTION: Shared state and helpers
// ====================================================================
// "My active patients" pool for the Rankings tab: a plain in-memory
// array, not localStorage, so it resets on page refresh by design.
// Patients are added here from a result card's "+ Add to active pool"
// button (predict.js) and read/ranked entirely client-side by the
// Rankings tab's own section (rankings.js) -- nothing here ever
// touches the backend.
let activePatients = []; // each: {label, risk_hepatic_event, risk_death, weighted_risk, fib4_score, apri_score, age_at_baseline, visit_count}

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

// A plain <td> with the given text.
function buildCell(text) {
    const cell = document.createElement("td");
    cell.textContent = text;
    return cell;
}
