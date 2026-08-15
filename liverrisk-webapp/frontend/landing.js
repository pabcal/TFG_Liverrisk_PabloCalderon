// ====================================================================
// SECTION: Landing screen -> app screen
// ====================================================================
// One-way transition: clicking "Start Predicting" hides the landing
// screen and reveals the app (header + tabs + all panels), which was
// already mounted in the DOM the whole time -- there's no path back to
// the landing screen from inside the app, so this is the only handler
// this file needs.
const landingScreen = document.querySelector("#landing-screen");
const appScreen = document.querySelector("#app-screen");
const startPredictingButton = document.querySelector("#start-predicting-button");

startPredictingButton.addEventListener("click", function () {
    landingScreen.classList.add("hidden");
    appScreen.classList.remove("hidden");
});
