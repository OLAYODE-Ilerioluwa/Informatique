// ======== INITIALISATION ========
document.addEventListener("DOMContentLoaded", () => {
  initSkillBars();
  // initForm(); // ← à activer si on ajoute la logique du formulaire
});

// ======== BARRES DE COMPÉTENCES ========
function initSkillBars() {
  const bars = document.querySelectorAll(".bar");
  bars.forEach(bar => {
    const level = parseInt(bar.getAttribute("data-level"), 10);
    const percent = (level / 3) * 100;

    const innerBar = document.createElement('div');
    innerBar.style.width = '0%';
    innerBar.style.height = '100%';
    innerBar.style.background = 'linear-gradient(90deg, #00c6ff, #0072ff)';
    innerBar.style.borderRadius = '10px';
    innerBar.style.transition = 'width 1.2s ease-out';

    bar.appendChild(innerBar);
    setTimeout(() => {
      innerBar.style.width = percent + '%';
    }, 100);
  });
}

// ======== FORMULAIRE DE CONTACT (optionnel) ========
function initForm() {
  const form = document.querySelector('.contact-form');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    alert('Merci pour votre message ! (fonctionnalité à compléter)');
    // Ici, tu pourrais envoyer les données via fetch() vers un back-end ou service externe
  });
}
