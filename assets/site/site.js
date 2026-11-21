(() => {
  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const toggle = document.getElementById('animation-toggle');
  const animations = [...document.querySelectorAll('[data-animation]')];
  const visible = new Set();
  let playing = !motionPreference.matches;
  let userSelected = false;
  function updateImage(img) {
    const src = playing && visible.has(img) ? img.dataset.animation : img.dataset.poster;
    if (img.getAttribute('src') !== src) img.src = src;
  }
  function updatePlayback() {
    toggle.textContent = playing ? 'Pause animations' : 'Play animations';
    toggle.setAttribute('aria-pressed', String(!playing));
    animations.forEach(updateImage);
  }
  toggle.hidden = false;
  toggle.addEventListener('click', () => { userSelected = true; playing = !playing; updatePlayback(); });
  motionPreference.addEventListener('change', () => {
    if (!userSelected) { playing = !motionPreference.matches; updatePlayback(); }
  });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) visible.add(entry.target); else visible.delete(entry.target);
        updateImage(entry.target);
      });
    }, { rootMargin: '120px' });
    animations.forEach(img => observer.observe(img));
  } else animations.forEach(img => visible.add(img));
  updatePlayback();

  const dialog = document.getElementById('figure-dialog');
  const expanded = document.getElementById('expanded-figure');
  if (typeof dialog.showModal === 'function') {
    document.querySelectorAll('[data-zoom]').forEach(button => {
      button.disabled = false;
      button.addEventListener('click', () => {
        const original = button.querySelector('img');
        expanded.src = original.src;
        expanded.alt = original.alt;
        dialog.showModal();
        document.body.classList.add('dialog-open');
      });
    });
    dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));
  }
})();
