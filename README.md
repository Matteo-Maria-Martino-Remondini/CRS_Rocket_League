# Classificazione di Skillshot in Rocket League tramite Random Forest

**Corso:** Intelligenza Artificiale 2 - Sapienza Università di Roma (A.A. 2025/2026)

**Gruppo di Lavoro:**
*   **Brando Cappucci** (Matricola: 2139484)
*   **Matteo Maria Martino Remondini** (Matricola: 2136808)
*   **Lorenzo Stranieri** (Matricola: 2153723)

---

## Obiettivo del Progetto

Questo progetto mira a sviluppare un classificatore di Machine Learning in grado di riconoscere automaticamente manovre acrobatiche complesse (*skillshot*) nel videogioco **Rocket League**.
Il sistema analizza dati di telemetria grezzi (serie temporali multivariate) per distinguere tra 7 classi di movimento (es. *Power Shot*, *Waving Dash*, *Air Dribble*), superando le sfide legate alla durata variabile delle sequenze e al rumore dei sensori.

## Punti di Forza e Metodologia

A differenza di approcci standard basati su modelli lineari, questo progetto introduce diverse innovazioni metodologiche:

1.  **Parsing Vettoriale Efficiente:** Invece di iterare le righe con cicli Python, abbiamo implementato un parser basato su operazioni vettoriali di Pandas, ottimizzando il caricamento e la strutturazione del dataset grezzo.
2.  **Feature Engineering "Fisico-Comportamentale":**
    *   **Robustezza:** Sostituzione di minimo/massimo con i **Quartili ($q_{25}, q_{75}$)** per filtrare gli outlier nei dati fisici (velocità, posizione).
    *   **Dinamica di Input:** Introduzione della metrica **"Toggle Count"** per misurare la frequenza di pressione dei tasti (es. "spamming" vs pressione prolungata), rivelatasi determinante per la classificazione.
3.  **Modellazione Non-Lineare:** Utilizzo di un **Random Forest Classifier** (selezionato dopo un confronto con SVM a Kernel RBF) per catturare le relazioni complesse tra le feature senza necessità di eccessiva riduzione dimensionale.

## Dataset

Il progetto utilizza il **Rocket League Skillshots Data Set** (UCI Machine Learning Repository).
*   **Struttura:** Serie temporali multivariate a lunghezza variabile.
*   **Classi:** 298 sequenze suddivise in 7 categorie (inclusa una classe "Noise").
*   **Acquisizione:** Il notebook scarica automaticamente i dati dalla fonte ufficiale UCI per garantire la riproducibilità.

## Pipeline del Progetto

Il workflow seguito nel notebook `IA2_Progetto_Cappucci_Remondini_Stranieri.ipynb` è il seguente:

1.  **Data Loading:** Download automatico e estrazione zip tramite librerie `urllib`.
2.  **Preprocessing:** Parsing vettoriale e pulizia dei dati.
3.  **Advanced EDA:** Analisi della distribuzione delle classi (Donut Chart) e della variabilità temporale (Box Plot stratificati).
4.  **Feature Engineering:** Generazione della matrice di feature $X$ (43 feature totali) usando statistiche robuste e conteggi di transizione.
5.  **Model Selection:** Torneo tra Random Forest e SVM tramite *Stratified 5-Fold Cross-Validation*.
6.  **Optimization:** Fine-tuning degli iperparametri del Random Forest (`n_estimators`, `min_samples_split`) tramite *GridSearchCV*.
7.  **Evaluation:** Valutazione finale su Test Set (20% dei dati), analisi della Matrice di Confusione e della Feature Importance.

## Risultati Chiave

Il modello finale ha raggiunto prestazioni eccellenti, superando le baseline standard:

*   **Accuracy sul Test Set:** **86.67%**
*   **Feature più Determinanti:**
    *   `DistanceBall_q25` (Posizione stabile della palla)
    *   `DistanceCeil_q25` (Altezza dal soffitto)
    *   `accelerate_count_toggles` (Frenesia sull'acceleratore)

L'analisi dell'importanza delle feature conferma che l'approccio basato sui quartili e sui conteggi di input è stato decisivo per il successo del modello.

## Requisiti e Utilizzo

Il progetto è sviluppato in Python 3. Le dipendenze principali sono:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn