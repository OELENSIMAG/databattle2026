# Prédiction fin d'orage — Data Battle 2026

## Contenu du dossier

docker/
├── Dockerfile ← recette de construction
├── requirements.txt ← librairies Python
├── api.py ← code de l'API
├── modele_fin_orage.json ← modèle XGBoost
├── label_encoder_airport.pkl ← encodeur des aéroports
└── feature_cols.pkl ← liste des features
└── storm-predictor.tar.gz ← Image Docker pré-construite (voir méthode 2)

---

## Installation de Docker (si pas encore fait)

Télécharge Docker Desktop : https://www.docker.com/products/docker-desktop/
Lance l'application et vérifie dans un terminal :
```
docker --version
```
---

## Méthode 1 — Construction à partir des sources

Le jury reconstruit l'image à partir des fichiers fournis.

### Étape 1 — Construire l'image
# Dans le dossier docker/
```
docker build -t storm-predictor .
````

### Étape 2 — Lancer le container
```
docker run -p 8000:8000 storm-predictor
````

### Étape 3 — Vérifier que ça tourne
```
curl http://localhost:8000/health
```

Réponse attendue :
```
{"status": "ok", "model": "modele_fin_orage.json"}
```

---

## Méthode 2 — Utiliser l'image pré-construite (alternative)

Si vous préférez utiliser directement l'image Docker sans reconstruire :

### ...
```
docker save storm-predictor | gzip > storm-predictor.tar.gz
```

### Charger l'image
```
docker load < storm-predictor.tar.gz
```

### Lancer le container
```
docker run -p 8000:8000 storm-predictor
```

---

## Utilisation — générer predictions.csv

Une fois le container lancé, dans un NOUVEAU terminal :

```
curl -X POST http://localhost:8000/predict \
     -F "file=@chemin/vers/fichier_de_test.csv" \
     --output predictions.csv
````

Le fichier predictions.csv est généré dans le dossier courant.


---

## Arrêter le container

# Trouver l'ID du container
docker ps

# L'arrêter
docker stop <ID>

Ou simplement Ctrl + C dans le terminal où tourne le container.



---

## FICHIERS FOURNIS

- storm-predictor.tar.gz : Image Docker pré-construite (alternative)
- Dossier docker/ : Sources complètes pour reconstruction

