document.querySelectorAll('nav a').forEach((el) => {
  el.addEventListener('mouseenter', () => { el.style.transform = 'translateY(-1px)'; });
  el.addEventListener('mouseleave', () => { el.style.transform = 'translateY(0)'; });
});

const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');

function resizePlots(scope) {
  if (!window.Plotly) return;
  const root = scope || document;
  root.querySelectorAll('.js-plotly-plot').forEach((plot) => {
    window.Plotly.Plots.resize(plot);
  });
}

if (tabButtons.length && tabPanels.length) {
  tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      tabButtons.forEach((b) => b.classList.remove('active'));
      tabPanels.forEach((p) => p.classList.remove('active'));

      btn.classList.add('active');
      const panel = document.getElementById(target);
      if (panel) {
        panel.classList.add('active');
        requestAnimationFrame(() => resizePlots(panel));
      }
    });
  });
}

window.addEventListener('resize', () => resizePlots(document));
window.addEventListener('load', () => resizePlots(document));
