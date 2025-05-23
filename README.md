https://www.youtube.com/watch?v=XqcDgiePl54&ab_channel=N%C3%A1ndorVincze


# Numszám Toolbox

A **numszam_toolbox** egy Python-alapú numerikus függvénykönyvtár, amely feladatokat generál és old meg különböző mátrixműveletekkel és numerikus algoritmusokkal. Használható oktatási célokra és gyakorláshoz, grafikus webes felülettel kiegészítve.


## Reports
A **reports** mappában megtalálható a **pylint** és a **flake8** elemzők, illetve a **pytest** eredménye is.

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
└── numszam_toolbox
    ├── decomp        # mátrix felbontások
    ├── iterative     # iterációs módszerek
    ├── poly          # polinom osztály
    ├── utils         # mátrix generálás, pretty printing
    └── web           # Flask webapp
        ├── problems  # feladat generálás
        ├── static    # .png, .css
        └── templates # .html

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


## Licensz

Ez a projekt az MIT licenc alatt érhető el. Szabadon másolható, módosítható és terjeszthető.

---

Bármilyen hibát, megjegyzést az nvincze@inf.u-szeged.hu-n várunk!
