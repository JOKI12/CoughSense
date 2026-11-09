# Méthodologie Scientifique & Traitement du Signal — CoughSense

**Projet de Master S3** : *FPGA, VHDL et Traitement du Signal*  
**Application** : *Analyse et classification des signaux de toux pour le dépistage expérimental des affections respiratoires*  

---

## 1. Cadre Mathématique du Traitement du Signal Audio

Le signal acoustique produit lors de l'expulsion bronchique est un phénomène non-stationnaire et transitoire résultant de l'interaction mécanique entre l'air comprimé pulmonaire, la muqueuse des voies aériennes supérieures et les cordes vocales.

### 1.1 Échantillonnage Discret et Théorème de Shannon-Nyquist

Soit $x_a(t)$ le signal acoustique continu de pression capté par le transducteur (microphone). L'échantillonnage temporel périodique à la période $T_s = \frac{1}{f_s}$ produit la suite discrète :

$$x[n] = x_a(n T_s), \quad n \in \mathbb{Z}$$

Selon le théorème d'échantillonnage de Nyquist-Shannon, pour éviter tout phénomène de repliement spectral (*aliasing*), la fréquence d'échantillonnage $f_s$ doit satisfaire :

$$f_s \ge 2 f_{max}$$

Dans **CoughSense**, nous fixons la fréquence standardisée à :

$$f_s = 16\,000\text{ Hz} \implies f_{Nyquist} = \frac{f_s}{2} = 8\,000\text{ Hz}$$

Cette bande passante utile $[0, 8000\text{ Hz}]$ couvre l'intégralité du spectre acoustique utile des bruits de toux chez l'adulte et l'enfant (généralement concentré entre 100 Hz et 4500 Hz).

---

## 2. Chaîne de Conditionnement & Prétraitement

### 2.1 Débruitage par Soustraction Spectrale (*Spectral Gating*)

Le bruit de fond ambiant est modélisé comme un processus aléatoire additif stationnaire au sens large :

$$y[n] = s[n] + b[n]$$

En passant dans le domaine de Fourier court-terme :

$$Y(m, k) = S(m, k) + B(m, k)$$

L'estimation du spectre de puissance du bruit $\hat{P}_b[k] = \mathbb{E}[|B(k)|^2]$ est obtenue sur les $10\%$ de trames d'énergie minimale (silence initial et pauses). Le spectre de magnitude débruité $|\hat{S}(m, k)|$ est alors calculé par soustraction non-linéaire avec plancher spectral :

$$|\hat{S}(m, k)| = \max\left( |Y(m, k)| - \alpha \sqrt{\hat{P}_b[k]}, \; \beta |Y(m, k)| \right)$$

où :
- $\alpha \approx 1.3$ est le facteur de sur-soustraction pour éliminer les résonances parasites.
- $\beta = 0.05$ est le plancher spectral (*spectral floor*) empêchant l'apparition d'artéfacts dits de "bruit musical".

Le signal temporel restauré $\hat{s}[n]$ est reconstruit via la Transformée de Fourier Inverse à Court-Terme (ISTFT) en préservant la phase d'origine $\theta_Y(m, k) = \arg(Y(m, k))$ :

$$\hat{s}[n] = \text{ISTFT}\left( |\hat{S}(m, k)| e^{j \theta_Y(m, k)} \right)$$

### 2.2 Filtrage Numérique Passe-Bande Butterworth (Phase Nulle)

Pour éliminer les bruits mécaniques infra-basses ($< 80\text{ Hz}$) et les composantes haute fréquence hors champ ($> 4000\text{ Hz}$), nous appliquons un filtre numérique à Réponse Impulsionnelle Infinie (IIR) de Butterworth d'ordre $N=4$ :

$$|H(j\omega)|^2 = \frac{1}{1 + \left( \frac{\omega^2 - \omega_0^2}{\omega \cdot B} \right)^{2N}}$$

Pour éliminer toute distorsion de phase qui modifierait la morphologie de l'onde de choc de la toux, le filtrage est appliqué de manière bidirectionnelle avant-arrière (*zero-phase forward-backward filtering*) :

$$y[n] = h[n] * (h[-n] * x[-n])$$

La fonction de transfert globale est $|H(e^{j\omega})|^2$, garantissant une phase strictement nulle :

$$\phi(\omega) = 0, \quad \forall \omega$$

### 2.3 Segmentation Énergétique par Enveloppe RMS

La toux se caractérise par trois phases physiologiques :
1. **Phase compressive/explosive** : Pic d'énergie bref et intense ($100-250\text{ ms}$).
2. **Phase intermédiaire/turbulente** : Souffle trachéal à large bande.
3. **Phase de relâchement vocal** : Décroissance de pression.

L'enveloppe énergétique RMS court-terme est calculée sur des fenêtres de $N_w = 512$ échantillons avec un saut de $H = 128$ :

$$\text{RMS}[m] = \sqrt{\frac{1}{N_w} \sum_{n=0}^{N_w-1} x^2[m H + n]}$$

Un seuil statistique adaptatif basé sur la médiane et l'écart absolu médian (MAD) isole automatiquement les régions d'émission :

$$\text{Seuil} = \text{Médiane}(\text{RMS}) + \lambda \cdot \text{MAD}(\text{RMS})$$

---

## 3. Analyse Temporelle & Fréquentielle

### 3.1 Descripteurs Temporels

- **Énergie RMS** :
  $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2[n]}$$
- **Taux de Passage par Zéro (Zero-Crossing Rate - ZCR)** :
  $$\text{ZCR} = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$
- **Facteur de Crête** :
  $$C = \frac{\max_{n} |x[n]|}{\text{RMS}}$$

### 3.2 Transformée de Fourier Rapide (FFT) & Paramètres Spectraux

Pour une séquence finie pondérée par une fenêtre de Hanning $w[n]$ :

$$X[k] = \sum_{n=0}^{N-1} x[n] w[n] e^{-j \frac{2\pi k n}{N}}, \quad k = 0, \dots, \frac{N}{2}$$

La fréquence physique associée au bin $k$ est $f_k = \frac{k \cdot f_s}{N}$.

- **Fréquence Dominante** :
  $$f_{dom} = \arg\max_{f_k} |X[k]|$$
- **Centroïde Spectral (Centre de gravité)** :
  $$f_c = \frac{\sum_k f_k |X[k]|}{\sum_k |X[k]|}$$
- **Largeur de Bande Spectrale (Dispersion)** :
  $$\sigma_f = \sqrt{\frac{\sum_k (f_k - f_c)^2 |X[k]|}{\sum_k |X[k]|}}$$
- **Rolloff Spectral ($85\%$)** : Fréquence $f_{roll}$ telle que :
  $$\sum_{k=0}^{K_{roll}} |X[k]|^2 = 0.85 \sum_{k} |X[k]|^2$$
- **Entropie Spectrale** : Mesure de désordre fréquentiel :
  $$H = -\frac{1}{\log_2(K)} \sum_{k=0}^{K-1} p_k \log_2(p_k), \quad p_k = \frac{|X[k]|^2}{\sum_j |X[j]|^2}$$
- **Platitude Spectrale (*Spectral Flatness*)** :
  $$\text{SF} = \frac{\exp\left( \frac{1}{K} \sum_{k} \ln |X[k]|^2 \right)}{\frac{1}{K} \sum_{k} |X[k]|^2}$$

---

## 4. Analyse Temps-Fréquence & Coefficients MFCC

### 4.1 Transformée de Fourier à Court-Terme (STFT)

$$X(m, k) = \sum_{n=0}^{N_w - 1} x[n + m H] w[n] e^{-j \frac{2\pi k n}{N_{FFT}}}$$

Le spectrogramme de puissance logarithmique (dB) s'exprime par :

$$S_{dB}(m, k) = 10 \log_{10} \left( \frac{|X(m, k)|^2}{\max(|X|^2) + \epsilon} \right)$$

### 4.2 Échelle Psychoacoustique de Mel & Coefficients MFCC

L'échelle de Mel modélise la résolution non-linéaire de l'oreille humaine :

$$m = 2595 \log_{10} \left( 1 + \frac{f}{700} \right)$$

Un banc de 64 filtres triangulaires chevauchants $H_m(k)$ est appliqué sur le spectre de puissance. L'énergie de chaque bande Mel est :

$$E_m = \sum_{k} |X(k)|^2 H_m(k)$$

La Transformée en Cosinus Discrète (DCT-II) est ensuite appliquée sur les log-énergies :

$$c_n = \sum_{m=0}^{M-1} \log(E_m) \cos\left( \frac{\pi n (m + 0.5)}{M} \right), \quad n = 1, \dots, 13$$

Pour capturer la cinétique de fermeture/ouverture du conduit vocal, les dérivées temporelles du premier ordre ($\Delta$) et second ordre ($\Delta\Delta$) sont calculées :

$$\Delta c_n[t] = \frac{\sum_{\tau=-d}^{d} \tau \cdot c_n[t + \tau]}{\sum_{\tau=-d}^{d} \tau^2}$$

Le vecteur complet regroupe **91 caractéristiques numériques** (7 temporelles, 6 spectrales, 26 MFCCs statiques, 26 Deltas, 26 Delta-Deltas).

---

## 5. Apprentissage Automatique & Prévention du Data Leakage

### 5.1 Protocole Anti-Fuite (*Participant Group Split*)

Pour éviter toute surévaluation artificielle due aux caractéristiques vocales propres à chaque individu, les sous-ensembles d'entraînement ($70\%$), validation ($15\%$) et test ($15\%$) sont partitionnés **par identifiant unique de participant** :

$$\text{Participant}_{Train} \cap \text{Participant}_{Test} = \emptyset$$

### 5.2 Modèles Implémentés

1. **SVM à noyau RBF** :
   $$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$
   Calibration probabiliste de Platt pour estimer $P(y=1|x)$.
2. **Forêt Aléatoire (Random Forest)** :
   Ensemble de 150 arbres de décision avec pondération équilibrée des classes.
3. **K-Plus Proches Voisins (KNN)** :
   Vote pondéré par l'inverse de la distance euclidienne normée ($k=5$).
4. **CNN 2D (Deep Learning)** :
   Réseau convolutif à 2 blocs Conv2D + BatchNorm + MaxPooling traitant le Mel-Spectrogramme normalisé $[64 \times 128]$.

---

## 6. Considérations Éthiques & Limites Scientifiques

1. **Portée Pédagogique et Non-Diagnostique** :
   Le dépistage par analyse spectrale seule ne remplace pas un examen clinique, une spirométrie ou un test PCR. Le système formule uniquement une probabilité de similarité acoustique (*"Pattern évocateur d'anomalie respiratoire"*).
2. **Variabilité du Matériel d'Enregistrement** :
   Les micros de smartphones induisent des distorsions harmoniques et des réponses en fréquence non-plates, atténuées en partie par notre filtrage passe-bande et la normalisation cepstrale.
