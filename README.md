# 🧠 Analyse de Sentiments Multiclasse — NLP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)

**Comparaison de modèles NLP pour la classification de sentiments en 3 classes**  
*du modèle classique au Transformer état de l'art*

[📓 Notebook](#-notebooks) · [📊 Résultats](#-résultats) · [🚀 Installation](#-installation) · [📁 Structure](#-structure-du-projet)

</div>

---

## 📌 Présentation du projet

Ce projet implémente un **pipeline NLP complet** pour classifier automatiquement des avis Amazon en trois catégories de sentiment :

| Classe | Label | Description |
|--------|-------|-------------|
| 😠 **Négatif** | 0 | Avis exprimant une insatisfaction |
| 😐 **Neutre** | 1 | Avis mitigé ou ambigu |
| 😊 **Positif** | 2 | Avis exprimant une satisfaction |

L'objectif central est **comparatif** : évaluer et analyser rigoureusement 4 approches de modélisation NLP, depuis les méthodes classiques (TF-IDF + ML) jusqu'aux architectures Transformer (DistilBERT).

---

## 🎯 Objectifs pédagogiques & scientifiques

- Maîtriser un **pipeline NLP de bout en bout** : collecte → prétraitement → modélisation → évaluation
- Comprendre les **avantages et limites** de chaque approche (classique vs. deep learning)
- Appliquer une **démarche rigoureuse d'évaluation** adaptée aux problèmes multiclasses déséquilibrés
- Produire un travail **reproductible et documenté**, prêt pour un contexte académique ou professionnel

---

## 📊 Résultats

### Tableau comparatif des modèles

| Rang | Modèle | Accuracy | F1-Macro | F1-Weighted |
|------|--------|----------|----------|-------------|
| 🥇 | SVM (LinearSVC) | 79.6% | 77.53% | 79.61% |
| 🥈 | Random Fores | % | 79.6% | 77.0% |79.45%
| 🥉 | Régression Logistique | 78.2% | 76.29% | 78.05% |
| 4️⃣ | DistilBERT (Zero-Shot) | 41.08% | 37.73% | 30.83% |

> ⚠️ **Note :** Les scores exacts varient selon le dataset et les paramètres. Ré-exécutez le notebook pour obtenir vos résultats précis.

### Visualisations produites

| Fichier | Contenu |
|---------|---------|
| `results/distribution_classes.png` | Distribution des 3 classes de sentiment |
| `results/wordclouds.png` | Nuages de mots par classe |
| `results/top_terms_tfidf.png` | Termes TF-IDF les plus discriminants |
| `results/confusion_matrices_classiques.png` | Matrices de confusion comparées |
| `results/comparaison_finale.png` | Graphique de comparaison finale |

---

## 🏗️ Architecture du pipeline

```
Données brutes (Amazon Reviews)
         │
         ▼
┌─────────────────────┐
│  1. Chargement      │  HuggingFace Datasets
│     & Labellisation │  Amazon Polarity → 3 classes
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. EDA             │  Distributions, longueurs,
│                     │  WordClouds, statistiques
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Prétraitement   │  Nettoyage → Tokenisation
│                     │  Stop words → Lemmatisation (spaCy)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Feature Eng.    │  TF-IDF (unigrammes + bigrammes)
│                     │  max_features=10 000
└────────┬────────────┘
         │
    ┌────┴──────────────────────────────┐
    │                                   │
    ▼                                   ▼
┌──────────────────┐         ┌──────────────────────┐
│  5. ML Classique │         │  6. Transformer       │
│  • Log. Reg.     │         │  • DistilBERT         │
│  • SVM           │         │  • Inférence directe  │
│  • Random Forest │         │  (zero-shot)          │
└────────┬─────────┘         └────────┬─────────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  7. Évaluation   │  Accuracy, F1-Macro,
         │     Comparaison  │  Matrices de confusion,
         │                  │  Analyse des erreurs
         └──────────────────┘
```

---

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip ou conda
- (Optionnel) GPU pour accélérer DistilBERT

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/votre-username/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### Étape 2 — Créer un environnement virtuel

```bash
# Avec venv
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Ou avec conda
conda create -n nlp-sentiment python=3.10
conda activate nlp-sentiment
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Étape 4 — Créer les dossiers nécessaires

```bash
mkdir -p results models
```

### Étape 5 — Lancer le notebook

```bash
jupyter notebook notebooks/sentiment_analysis_complet.ipynb
```

---

## 📁 Structure du projet

```
sentiment-analysis-nlp/
│
├── sentiment_analysis_complet.ipynb    # Notebook principal 
│
│
├── 📊 results/                              # Figures et visualisations (auto-générées)
│   ├── distribution_classes.png
│   ├── wordclouds.png
│   ├── top_terms_tfidf.png
│   ├── confusion_matrices_classiques.png
│   ├── comparaison_finale.png
│   └── resultats_comparaison.csv
│
├── 🤖 models/                              # Modèles sauvegardés (auto-générés)
│   ├── tfidf_vectorizer.pkl
│   └── svm_model.pkl
│
├── 📄 requirements.txt                     # Dépendances Python
├── 📄 README.md                            # Ce fichier
```

---

## 🔬 Méthodologie détaillée

### Données

- **Source :** [Amazon Polarity Dataset](https://huggingface.co/datasets/amazon_polarity) via HuggingFace
- **Taille utilisée :** 5 000 exemples (extensible)
- **Langue :** Anglais
- **Split :** 80% train / 20% test (stratifié)

### Prétraitement

Le pipeline de prétraitement appliqué à chaque texte :

1. **Mise en minuscules**
2. **Suppression** du HTML, des URLs et des caractères non-alphabétiques
3. **Tokenisation** avec spaCy
4. **Suppression des stop words** (liste intégrée spaCy)
5. **Lemmatisation** — `running` → `run`, `better` → `good`
6. **Filtrage** des tokens de moins de 3 caractères

### Feature Engineering

- **TF-IDF** avec bigrammes (`ngram_range=(1,2)`), 10 000 features max
- Normalisation `sublinear_tf=True` (log-TF)
- Filtrage `min_df=3`, `max_df=0.85`

### Évaluation

La métrique principale est le **F1-Score Macro**, car il traite toutes les classes de façon équitable, ce qui est adapté aux classes potentiellement déséquilibrées.

---

## 💡 Points techniques clés

- **Reproductibilité** : `random_state=42` partout, split stratifié
- **Modularité** : chaque étape est une fonction documentée et réutilisable
- **Persistance** : les modèles sont sauvegardés avec `pickle` pour une inférence future
- **Efficacité** : inférence DistilBERT par batch pour optimiser le temps de calcul

---

## 🔭 Perspectives d'amélioration

- **Fine-tuning de CamemBERT** sur un dataset français pour une version multilingue
- **Optimisation des hyperparamètres** avec `GridSearchCV` ou `Optuna`
- **Data augmentation** pour équilibrer les classes
- **Déploiement** via une API FastAPI + interface Streamlit
- **Intégration MLflow** pour le tracking des expériences

---

## 📚 Références

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [spaCy Documentation](https://spacy.io/usage)
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

---

## 👤 Auteur

**NGOM Khadim**  
Étudiant Master 1 Informatique   
🔗 [LinkedIn](www.linkedin.com/in/khadim-ngom-65825728b)

---

## 📄 Licence

Projet académique — Usage éducatif uniquement.


