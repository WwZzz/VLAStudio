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
window.VLAView = {
  active: () => !document.hidden &&
    (window.parent === window || !window.parent.VLAWorkspace || window.parent.VLAWorkspace.isActive(window)),
};
if (window.parent !== window && window.parent.VLAWorkspace)
  document.documentElement.classList.add('workspace-embedded');
(() => {
  let invalidated = false;
  function update() {
    if (invalidated && window.VLAView.active()) {
      invalidated = false;
      document.dispatchEvent(new Event('vla-settingsrefresh'));
    }
    document.dispatchEvent(new Event('vla-viewvisibility'));
  }
  document.addEventListener('vla-settingsinvalidated', () => { invalidated = true; update(); });
  document.addEventListener('vla-viewchange', update);
  document.addEventListener('visibilitychange', update);
  window.addEventListener('storage', event => {
    if (event.key === 'vla-language' || event.key === 'vla-theme')
      document.dispatchEvent(new Event('vla-preferenceschange'));
  });
})();
