function initNavHover() {
  document.querySelectorAll('nav a').forEach((el) => {
    el.addEventListener('mouseenter', () => { el.style.transform = 'translateY(-1px)'; });
    el.addEventListener('mouseleave', () => { el.style.transform = 'translateY(0)'; });
  });
}

function resizePlots(scope) {
  if (!window.Plotly) return;
  const root = scope || document;
  root.querySelectorAll('.js-plotly-plot').forEach((plot) => {
    window.Plotly.Plots.resize(plot);
  });
}

function initTabs(scope) {
  const root = scope || document;
  const tabButtons = root.querySelectorAll('.tab-btn');
  const tabPanels = root.querySelectorAll('.tab-panel');
  if (!tabButtons.length || !tabPanels.length) return;

  tabButtons.forEach((btn) => {
    if (btn.dataset.tabBound === '1') return;
    btn.dataset.tabBound = '1';
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      tabButtons.forEach((b) => b.classList.remove('active'));
      tabPanels.forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = root.querySelector(`#${target}`);
      if (panel) {
        panel.classList.add('active');
        requestAnimationFrame(() => resizePlots(panel));
      }
    });
  });
}

function initMlLoader() {
  const content = document.querySelector('[data-ml-content]');
  const loader = document.querySelector('[data-ml-loader]');
  const retry = document.querySelector('[data-ml-retry]');
  const steps = loader ? Array.from(loader.querySelectorAll('[data-ml-step]')) : [];
  const progress = loader ? loader.querySelector('.ml-bar') : null;
  let stepTimer = null;
  if (!content || !loader) return;

  const setStepState = (activeIndex, errorText) => {
    steps.forEach((step, index) => {
      const state = step.querySelector('.ml-step-state');
      step.classList.remove('is-active', 'is-done');
      if (index < activeIndex) {
        step.classList.add('is-done');
        if (state) state.textContent = 'terminé';
      } else if (index === activeIndex) {
        step.classList.add('is-active');
        if (state) state.textContent = errorText || 'en cours';
      } else {
        if (state) state.textContent = 'en attente';
      }
    });
    if (progress && steps[activeIndex]) {
      const name = steps[activeIndex].querySelector('.ml-step-name');
      const label = name ? name.textContent : 'Chargement en cours';
      progress.setAttribute('aria-valuetext', label);
    }
  };

  const advanceSteps = () => {
    if (!steps.length) return;
    let activeIndex = 0;
    setStepState(activeIndex);
    if (stepTimer) clearInterval(stepTimer);
    stepTimer = setInterval(() => {
      if (activeIndex < steps.length - 1) {
        activeIndex += 1;
        setStepState(activeIndex);
      } else {
        clearInterval(stepTimer);
      }
    }, 2600);
    return () => {
      if (stepTimer) clearInterval(stepTimer);
      stepTimer = null;
      if (steps.length) {
        setStepState(steps.length - 1);
        steps.forEach((step) => {
          step.classList.remove('is-active');
          step.classList.add('is-done');
          const state = step.querySelector('.ml-step-state');
          if (state) state.textContent = 'done';
        });
      }
    };
  };

  const executeScripts = (root) => {
    root.querySelectorAll('script').forEach((oldScript) => {
      const newScript = document.createElement('script');
      if (oldScript.src) {
        newScript.src = oldScript.src;
      } else {
        newScript.textContent = oldScript.textContent;
      }
      document.body.appendChild(newScript);
      oldScript.remove();
    });
  };

  const run = () => {
    loader.classList.remove('is-hidden');
    loader.classList.remove('has-error');
    loader.setAttribute('aria-busy', 'true');
    if (retry) retry.classList.remove('is-visible');
    const finalizeSteps = advanceSteps();

    fetch('/ml-pack', { headers: { 'X-Requested-With': 'fetch' } })
      .then((response) => {
        if (!response.ok) throw new Error('ml-pack failed');
        return response.text();
      })
      .then((html) => {
        const temp = document.createElement('div');
        temp.innerHTML = html;
        content.innerHTML = '';
        while (temp.firstChild) {
          content.appendChild(temp.firstChild);
        }
        executeScripts(content);
        loader.classList.add('is-hidden');
        loader.setAttribute('aria-busy', 'false');
        if (finalizeSteps) finalizeSteps();
        initTabs(content);
        resizePlots(content);
      })
      .catch(() => {
        const sub = loader.querySelector('.ml-loader-sub');
        if (sub) {
          sub.textContent = 'Echec du chargement. Le modele tourne peut-etre encore.';
        }
        if (steps.length) {
          setStepState(Math.max(0, steps.length - 1), 'erreur');
        }
        loader.classList.add('has-error');
        loader.setAttribute('aria-busy', 'false');
        if (retry) retry.classList.add('is-visible');
      });
  };

  if (retry) {
    retry.addEventListener('click', () => run());
  }

  run();
}

document.addEventListener('DOMContentLoaded', () => {
  initNavHover();
  initTabs(document);
  initMlLoader();
});

window.addEventListener('resize', () => resizePlots(document));
window.addEventListener('load', () => resizePlots(document));
