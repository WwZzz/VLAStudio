"use strict";
// Run synchronously in <head>, before stylesheets and the first page paint.
window.VLATheme = {
  apply() {
    let saved;
    try { saved = localStorage.getItem("vla-theme"); } catch {}
    const theme = saved === "dark" || saved === "light" ? saved
      : matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.dataset.theme = theme;
    return theme;
  }
};
window.VLATheme.apply();
