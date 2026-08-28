# -*- coding: utf-8 -*-
"""
Testes do pipeline sem banco: base sintética -> df_long.

Rode depois de mexer em limpeza, cluster, agregação ou upsert:

    python analise/acionador/testes_pipeline.py

Cobre o que quebra silencioso — colapso por listagem, dedupe, percentis
consistentes, índice repeat, e o payload do upsert bater com as colunas do
INSERT. Não toca no banco.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acionador import EstudoMercado  # noqa: E402


def _base_sintetica(n_imoveis: int = 400, n_duplicados: int = 30,
                    deriva_semanal: float = 0.001, seed: int = 42) -> pd.DataFrame:
    """
    Painel semanal artificial com deriva de preço conhecida e duplicatas
    plantadas — o mesmo imóvel físico anunciado com outro código.
    """
    rng = np.random.default_rng(seed)
    semanas = [date(2026, 3, 1) + timedelta(days=7 * i) for i in range(13)]

    imoveis = []
    for i in range(n_imoveis):
        imoveis.append(dict(
            codigo=f"C{i:04d}",
            area_util=float(rng.integers(60, 600)),
            m2=max(float(rng.normal(11000, 2500)), 3000.0),
            quartos=int(rng.integers(1, 5)),
            vagas=int(rng.integers(0, 3)),
            lat=round(float(rng.normal(-15.8, 0.02)), 6),
            lon=round(float(rng.normal(-47.9, 0.02)), 6),
            quadra=f"QI {int(rng.integers(1, 30))}",
            semanas_no_ar=int(rng.integers(2, 13)),
        ))
    for j in range(n_duplicados):
        dup = dict(imoveis[j])
        dup["codigo"] = f"D{j:04d}"
        imoveis.append(dup)

    linhas = []
    for im in imoveis:
        for k, sem in enumerate(semanas[: im["semanas_no_ar"]]):
            m2 = im["m2"] * (1 + deriva_semanal * k)
            linhas.append(dict(
                codigo=im["codigo"], bairro="LAGO SUL", cidade="BRASILIA",
                tipo="CASA", oferta="VENDA", area_util=im["area_util"],
                preco=m2 * im["area_util"], quartos=im["quartos"],
                vagas=im["vagas"], latitude=im["lat"], longitude=im["lon"],
                quadra=im["quadra"], data_coleta=sem,
            ))
    return pd.DataFrame(linhas)


def _pipeline(df_raw: pd.DataFrame, **kwargs):
    em = EstudoMercado(bairro="LAGO SUL", tipo="CASA", meses_alvo=["2026-05"], **kwargs)
    limpo = em._limpar_dados(df_raw)
    scaler = km = mapping = None
    try:
        scaler, km, mapping = em._treinar_cluster_global(limpo, "LAGO SUL")
    except Exception as e:
        print(f"[WARN] cluster off: {e}")
    long = em._build_metricas_long(
        limpo, "2026-05", date(2026, 3, 1), date(2026, 5, 31),
        "LAGO SUL", "CASA", scaler, km, mapping,
    )
    return em, limpo, long


def teste_limpeza(df_raw):
    em, limpo, _ = _pipeline(df_raw)
    assert not limpo.empty
    # colapso: no máximo uma linha por código/mês
    assert limpo.duplicated(subset=["codigo", "mes_ref"]).sum() == 0, "colapso falhou"
    # dedupe: os códigos D* plantados não podem sobreviver junto com os C*
    assert limpo["codigo"].nunique() < df_raw["codigo"].nunique(), "dedupe não removeu nada"
    print("OK  limpeza (dedupe + colapso)")


def teste_colunas_e_percentis(df_raw):
    _, _, long = _pipeline(df_raw)
    esperadas = ["amostra", "imoveis_unicos", "m2_p25", "m2_p75", "m2_desvio",
                 "variacao_m2_pct", "variacao_repeat_pct", "n_repeat", "confiabilidade"]
    faltando = [c for c in esperadas if c not in long.columns]
    assert not faltando, f"colunas ausentes: {faltando}"
    ok = ((long["m2_p25"] <= long["m2_mediana"] + 1e-6)
          & (long["m2_mediana"] <= long["m2_p75"] + 1e-6))
    assert ok.all(), "p25 <= mediana <= p75 violado"
    print(f"OK  colunas e percentis ({len(long)} linhas geradas)")


def teste_indice_repeat(df_raw):
    """
    A base tem deriva real de ~+0,4%/mês. `variacao_repeat_pct` tem que ficar
    perto disso; `variacao_m2_pct` pode divergir muito — é justamente o efeito
    de composição que o índice repeat existe para remover.
    """
    _, _, long = _pipeline(df_raw)
    rep = long["variacao_repeat_pct"].dropna()
    assert not rep.empty, "nenhum par de meses consecutivos encontrado"
    mediana = rep.median()
    assert -2.0 < mediana < 2.0, f"repeat fora do esperado: {mediana}"
    mix = long["variacao_m2_pct"].dropna()
    print(f"OK  índice repeat (repeat={mediana:+.2f}%  vs  mix={mix.median():+.2f}%)")


def teste_cauda(df_raw):
    n_winsor = len(_pipeline(df_raw)[1])
    n_iqr = len(_pipeline(df_raw, metodo_outlier="iqr")[1])
    n_none = len(_pipeline(df_raw, metodo_outlier="none")[1])
    assert n_winsor == n_none, "winsor não pode remover linhas"
    assert n_iqr <= n_winsor, "iqr deveria remover linhas"
    print(f"OK  cauda (winsor={n_winsor}  iqr={n_iqr}, descarta {n_winsor - n_iqr})")


def teste_payload_upsert(df_raw):
    """
    As colunas do INSERT são declaradas dentro de `_upsert`; se o pipeline
    parar de produzir uma delas, só se descobre em produção. Aqui se compara
    a lista real do arquivo com o df_long.
    """
    _, _, long = _pipeline(df_raw)
    fonte = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "acionador.py"), encoding="utf-8").read()
    bloco = fonte.split("        cols = [", 1)[1].split("]", 1)[0]
    cols = [c.strip().strip('",') for c in bloco.replace("\n", " ").split()]
    cols = [c for c in cols if c]
    faltando = [c for c in cols if c not in long.columns]
    assert not faltando, f"o INSERT pede colunas que o pipeline não gera: {faltando}"

    bloco_df = long[cols].astype(object)
    bloco_df = bloco_df.where(bloco_df.notna(), None)
    payload = list(bloco_df.itertuples(index=False, name=None))
    assert all(len(t) == len(cols) for t in payload)
    # psycopg2 não adapta numpy.int64/float64: o payload tem que sair em tipos nativos
    modulos = {type(v).__module__ for t in payload for v in t}
    assert modulos <= {"builtins", "datetime"}, f"tipos não adaptáveis: {modulos}"
    print(f"OK  payload do upsert ({len(payload)} tuplas x {len(cols)} colunas)")


if __name__ == "__main__":
    df_raw = _base_sintetica()
    print(f"base sintética: {len(df_raw)} linhas, {df_raw['codigo'].nunique()} códigos\n")
    teste_limpeza(df_raw)
    teste_colunas_e_percentis(df_raw)
    teste_indice_repeat(df_raw)
    teste_cauda(df_raw)
    teste_payload_upsert(df_raw)
    print("\nTODOS OS TESTES PASSARAM")
