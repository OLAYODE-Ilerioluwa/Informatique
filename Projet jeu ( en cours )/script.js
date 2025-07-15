const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const WIDTH = canvas.width;
const HEIGHT = canvas.height;

// === Chargement des 3 couches ===
const sky = new Image();
const clouds = new Image();
const foreground = new Image();

sky.src = 'images/sky_layer.png';
clouds.src = 'images/clouds_layer.png';
foreground.src = 'images/foreground_layer.png';

let cloudX = 0;

// Attendre que tout soit chargé avant d’animer
Promise.all([
  new Promise(res => sky.onload = res),
  new Promise(res => clouds.onload = res),
  new Promise(res => foreground.onload = res)
]).then(() => {
  requestAnimationFrame(update);
});

function update() {
  drawScene();
  requestAnimationFrame(update);
}

function drawScene() {
  // 1. Ciel fixe
  ctx.drawImage(sky, 0, 0, WIDTH, HEIGHT);

  // 2. Nuages animés
  cloudX -= 0.4;
  if (cloudX <= -WIDTH) cloudX = 0;
  ctx.drawImage(clouds, cloudX, 0, WIDTH, HEIGHT);
  ctx.drawImage(clouds, cloudX + WIDTH, 0, WIDTH, HEIGHT);

  // 3. Premier plan (dojo et arbres)
  ctx.drawImage(foreground, 0, 0, WIDTH, HEIGHT);
}
// === Personnage ===
const samurai = new Image();
samurai.src = 'samurai.png';

let samuraiX = 300; // position horizontale de départ
let samuraiY = 280; // position verticale fixe (sol)
let speed = 4;
let keys = {};

// Écoute clavier
document.addEventListener('keydown', e => keys[e.key] = true);
document.addEventListener('keyup', e => keys[e.key] = false);

// Affichage du samouraï
function drawSamurai() {
  // Gestion mouvement
  if (keys["ArrowLeft"]) samuraiX -= speed;
  if (keys["ArrowRight"]) samuraiX += speed;

  // Limites
  samuraiX = Math.max(0, Math.min(WIDTH - 50, samuraiX));

  // Dessin
  ctx.drawImage(samurai, samuraiX, samuraiY, 50, 70); // image, x, y, largeur, hauteur
}

// Mets ça dans la boucle
function update() {
  drawScene();
  drawSamurai(); // <-- ajoute le perso ici
  requestAnimationFrame(update);
}
