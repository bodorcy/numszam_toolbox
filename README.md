
# Numszám Toolbox

A **numszam_toolbox** egy Python-alapú numerikus függvénykönyvtár, amely feladatokat generál és old meg különböző mátrixműveletekkel és numerikus algoritmusokkal. Használható oktatási célokra és gyakorláshoz, grafikus webes felülettel kiegészítve.

## Funkciók

- LU-felbontás
- QR-felbontás
- Cholesky-felbontás
- Gauss–Seidel iteráció
- Hatványmódszer (legnagyobb sajátérték közelítése)
- Lagrange-polinom interpoláció (grafikonnal)
- Webes interfész Flask segítségével

## Projekt felépítése

```
numszam_toolbox/
├── src/
│   ├── poly/               # Polinom reprezentációk és műveletek
│   ├── decomp/             # Mátrixfelbontási algoritmusok
│   ├── iterative/          # Iterációs módszerek
│   ├── problems/           # Feladatgenerátorok
│   ├── web/                # Flask alkalmazás
│   └── utils.py            # Segédfüggvények
├── tests/                  # Tesztek
├── templates/              # HTML sablonok (Flask)
├── static/                 # Statikus fájlok (grafikon, CSS)
└── requirements.txt        # Függőségek listája

```

## Telepítés

1. **Függőségek telepítése**:

```bash
git clone https://github.com/bodorcy/numszam_toolbox.git
cd numszam_toolbox
pip install -r requirements.txt
```

2. **Telepítés csomagként**:

```bash
pip install .
```

## Webalkalmazás használata

A Flask webes felület indításához:

```bash
cd src/web
flask run
```

Ezután megnyitható a böngészőben:

[http://127.0.0.1:5000](http://127.0.0.1:5000)


## Licenc

Ez a projekt az MIT licenc alatt érhető el. Szabadon másolható, módosítható és terjeszthető.

---

Bármilyen hibát, megjegyzést az nvincze@inf.u-szeged.hu-n várunk!
