
# 🧮 Numszám Toolbox

A **numszam_toolbox** egy Python-alapú numerikus lineáris algebrai eszköztár, amely véletlenszerű feladatokat generál és old meg különböző mátrixműveletekkel és numerikus algoritmusokkal. Használható oktatási célokra, gyakorláshoz vagy akár demonstrációhoz is, grafikus webes felülettel kiegészítve.

## 🎯 Funkciók

- LU-felbontás
- QR-felbontás
- Cholesky-felbontás
- Gauss–Seidel iteráció
- Hatványmódszer (legnagyobb sajátérték közelítése)
- Lagrange-polinom interpoláció (grafikonnal)
- Webes interfész Flask segítségével

## 🗂 Projekt felépítése

```
numszam_toolbox/
├── src/
│   ├── poly/               # Polinom reprezentációk és műveletek
│   ├── decomp/             # Mátrixfelbontási algoritmusok
│   ├── iterative/          # Iteratív megoldási módszerek
│   ├── problems/           # Feladatgenerátorok
│   ├── web/                # Flask alkalmazás
│   └── utils.py            # Segédfüggvények
├── tests/                  # Egységtesztek
├── templates/              # HTML sablonok (Flask)
├── static/                 # Statikus fájlok (grafikonok, CSS)
├── requirements.txt        # Függőségek listája
├── pyproject.toml          # Csomagolási metaadatok
├── README.md               # Ez a fájl
└── LICENSE                 # Licenc (MIT)
```

## ⚙️ Telepítés

1. **Függőségek telepítése**:

```bash
git clone https://github.com/bodorcy/numszam_toolbox.git
cd numszam_toolbox
pip install -r requirements.txt
```

2. **(Opcionális) Telepítés csomagként**:

```bash
pip install .
```

## 🌐 Webalkalmazás használata

A Flask webes felület indításához:

```bash
cd src/web
flask run
```

Ezután megnyithatod a böngészőben:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

### Választható témakörök:
- LU
- QR
- Cholesky
- Gauss–Seidel
- Hatványmódszer
- Lagrange-polinom (grafikus kirajzolással)

## 🧪 Tesztelés

Az összes teszt futtatása:

```bash
pytest
```

Tesztlefedettség ellenőrzése:

```bash
pytest --cov=src
```

## ✅ Kódellenőrzés

A projekt megfelel a `flake8` és `pylint` által elvárt konvencióknak:

```bash
flake8 .
pylint src/
```

## 📝 Példa Lagrange feladatra

A rendszer képes interpolációs polinomokat illeszteni:

```text
Alappontok:
[(-4, 3), (-1, -2), (2, 5)]

Lagrange polinom:
0.50x^2 + 1.00x + 2.00

Részpolinomok:
L1 = ...
L2 = ...
L3 = ...
```

Az eredményt egy `.png` formátumú grafikonként is elmenti a `static/` mappába.

## 📦 Csomagolás fejlesztőknek

Ez a projekt használja a `pyproject.toml` szabványos konfigurációt. A csomagolás és terjesztés előtt:

```bash
python -m build
```

## 📄 Licenc

Ez a projekt az MIT licenc alatt érhető el. Szabadon másolható, módosítható és terjeszthető.

---

### ✨ Közreműködés

Szívesen fogadunk hibajelentéseket és javaslatokat! Nyiss egy `issue`-t vagy készíts `pull request`-et a [GitHub repóban](https://github.com/bodorcy/numszam_toolbox).