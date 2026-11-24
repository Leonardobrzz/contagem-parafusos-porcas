"""Conta parafusos e porcas em fotos de uma bancada (fundo de papel branco).

Uso:
    python contar.py fotos/Foto_1_parafuso_porca.jpg
    python contar.py fotos/ --referencia referencia_manual.csv --saida resultados/
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

LARGURA = 1500   # largura depois de redimensionar
K_FUNDO = 75     # EE pra estimar o fundo: maior que a largura das peças
K_AB = 3         # EE da abertura: tira pontinhos sem apagar as porcas pequenas
K_FE = 5         # EE do fechamento: tapa brilho e rachadura dentro da peça
AREA_MIN = 300   # menor peça aceita (px)


def ee(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def preencher(m):
    mp = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cheia = mp.copy()
    aux = np.zeros((mp.shape[0] + 2, mp.shape[1] + 2), np.uint8)
    cv2.floodFill(cheia, aux, (0, 0), 255)
    furos = cv2.bitwise_not(cheia)[1:-1, 1:-1]
    return cv2.bitwise_or(m, furos)


def pecas(m):
    n, _, est, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape
    lista = []
    for i in range(1, n):
        x, y, w, h, a = est[i]
        if a < AREA_MIN:
            continue
        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        lista.append((x, y, w, h, a))
    return lista


def classificar(lista, img):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    parafusos, porcas = 0, 0
    for x, y, w, h, a in lista:
        razao = w / h
        ocupa = a / (w * h)
        if 0.8 <= razao <= 1.25 and ocupa >= 0.55:
            porcas += 1
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 4)  # azul = porca
        else:
            parafusos += 1
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)  # vermelho = parafuso
    return parafusos, porcas, vis


def pipeline(caminho):
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(caminho)
    esc = LARGURA / img.shape[1]
    img = cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(img, (5, 5), 0)
    fundo = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ee(K_FUNDO))
    r = cv2.divide(g, fundo, scale=255)
    _, m = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ee(K_AB))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ee(K_FE))
    m = preencher(m)
    par, por, vis = classificar(pecas(m), img)
    return par, por, m, vis


def ler_referencia(caminho):
    with open(caminho, newline='', encoding='utf-8') as f:
        return {l['foto']: (int(l['parafusos']), int(l['porcas'])) for l in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('entrada', help='foto ou pasta de fotos')
    ap.add_argument('--referencia', help='CSV com a contagem manual (foto,parafusos,porcas)')
    ap.add_argument('--saida', help='pasta pra salvar as fotos com as caixas desenhadas')
    args = ap.parse_args()

    entrada = Path(args.entrada)
    fotos = sorted(entrada.glob('*.jpg'), key=lambda p: int(p.stem.split('_')[1])) if entrada.is_dir() else [entrada]
    ref = ler_referencia(args.referencia) if args.referencia else {}
    if args.saida:
        Path(args.saida).mkdir(parents=True, exist_ok=True)

    print(f"{'foto':<28} {'parafusos':>9} {'porcas':>7}" + ('   ref (par/porca)' if ref else ''))
    erros = []
    for f in fotos:
        par, por, _, vis = pipeline(f)
        linha = f'{f.name:<28} {par:>9} {por:>7}'
        if f.name in ref:
            rp, rq = ref[f.name]
            linha += f'   {rp}/{rq}'
            erros.append(abs(par - rp) + abs(por - rq))
        print(linha)
        if args.saida:
            cv2.imwrite(str(Path(args.saida) / f.name), vis)
    if erros:
        exatas = sum(e == 0 for e in erros)
        print(f'\nfotos com contagem exata: {exatas}/{len(erros)}')


if __name__ == '__main__':
    main()
