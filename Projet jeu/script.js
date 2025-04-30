
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const background = new Image();
background.src ='images/background.png'; // adapte le chemin selon ton projet

const player1 = {
  x: 150,
  y: 200,
  color: '#3498db',
  action: '',
  power: 0
};

const player2 = {
  x: 550,
  y: 200,
  color: '#e67e22',
  action: '',
  power: 0
};

const keys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '];
const keyLabels = {
  'ArrowUp': 'Attaque forte',
  'ArrowDown': 'Attaque rapide',
  'ArrowLeft': 'Esquive gauche',
  'ArrowRight': 'Esquive droite',
  ' ': 'Garde'
};

let roundInProgress = false;

function drawPlayer(player) {
  ctx.fillStyle = player.color;
  ctx.fillRect(player.x, player.y, 50, 100);
}

function drawScene() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  ctx.drawImage(background, 0, 0, canvas.width, canvas.height);
  
  drawPlayer(player1);
  drawPlayer(player2);
}

function resolveRound() {
  const resultText = document.getElementById('resultText');
  if (!player1.action || !player2.action) {
    resultText.textContent = "Les deux joueurs doivent jouer !";
    return;
  }

  resultText.textContent = `J1: ${keyLabels[player1.action]} | J2: ${keyLabels[player2.action]}`;

  // Reset actions
  player1.action = '';
  player2.action = '';
  roundInProgress = false;
}

document.addEventListener('keydown', (e) => {
  if (!roundInProgress && keys.includes(e.key)) {
    if (!player1.action) {
      player1.action = e.key;

      // L’ordi réagit automatiquement
      const randomKey = keys[Math.floor(Math.random() * keys.length)];
      player2.action = randomKey;

      roundInProgress = true;
      setTimeout(resolveRound, 1000);
    }
  }
});


drawScene();

function updateHealthBars() {
  document.getElementById('hp1').style.width = `${player1.hp}%`;
  document.getElementById('hp2').style.width = `${player2.hp}%`;
}

function flashPlayer(playerIndex) {
  const canvas = document.getElementById('gameCanvas');
  canvas.classList.add('flash-hit');
  setTimeout(() => {
    canvas.classList.remove('flash-hit');
  }, 200);
}

function resolveRound() {
  const resultText = document.getElementById('resultText');
  if (!player1.action || !player2.action) {
    resultText.textContent = "Les deux joueurs doivent jouer !";
    return;
  }

  resultText.textContent = `J1: ${keyLabels[player1.action]} | J2: ${keyLabels[player2.action]}`;

  // Démo simple : on dit que player1 touche player2
  if (Math.random() > 0.5) {
    player2.hp -= 10;
    flashPlayer(2);
  } else {
    player1.hp -= 10;
    flashPlayer(1);
  }

  updateHealthBars();

  // Reset
  player1.action = '';
  player2.action = '';
  roundInProgress = false;
}
