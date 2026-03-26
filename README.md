<h1>
	<img src="infomation/images/logo_databattle2026.png" alt="Logo DataBattle 2026" width="56" style="vertical-align: middle;" />
	Data Battle 2026
</h1>

Projet Data Battle 2026: analyse de donnees d'orages et prediction de fin d'alerte.

## Quick Start

### 1. Creer l'environnement virtuel

Depuis la racine du projet:

```bash
python3 -m venv .venv
```

Activation sur macOS/Linux:

```bash
source .venv/bin/activate
```

Activation sur Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Installer les dependances

```bash
pip install -r requirements.txt
```

### 3. Lancer les notebooks

- `src/notebook/analyse_data.ipynb` : analyse de donnees
- `src/notebook/model.ipynb` : modele predictif

### 4. Lancer la demo Streamlit

```bash
cd src
streamlit run app.py
```

Puis ouvrir l'URL locale affichee dans le terminal.

## Use Case

![Use Case](infomation/images/use_case.png)

<h2>
	<img src="infomation/images/Grenoble%20INP%20-%20Logo.png" alt="Grenoble INP Logo" width="120" style="vertical-align: middle;" />
	Equipe
</h2>

Nom de l'equipe: `Les marins d'eau douce`

| Membre | Email |
|---|---|
| EL OUADIFI Othmane | othmane.el-ouadifi@grenoble-inp.org |
| OUKHTITE Omar | omar.oukhtite@grenoble-inp.org |
| IDBRAYME Omar | omar.idbrayme@grenoble-inp.org |
| CHLIHI Mohamed Ziyad | mohamed-ziyad.chlihi@grenoble-inp.org |