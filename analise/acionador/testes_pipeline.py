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
    # Mesmo caminho do _rodar_pipeline: lagos usam cluster por faixa de luxo.
    cortes = em._treinar_cluster_por_faixa(limpo, "LAGO SUL/CASA")
    long = em._build_metricas_long(
        limpo, "2026-05", date(2026, 3, 1), date(2026, 5, 31),
        "LAGO SUL", "CASA", cortes_faixa=cortes,
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


def teste_agrupamento_quadras():
    """
    Cada bairro tem sua regra de quadra e elas não podem se atropelar:
    lagos viram Início/Meio/Final, Asa Norte e Asa Sul viram centena, e
    bairro sem regra mantém a quadra crua.
    """
    em = EstudoMercado()
    casos = [
        ("ASA NORTE", "SQN 409 BLOCO L", "SQN 400"),
        ("ASA NORTE", "SQN 116", "SQN 100"),
        ("ASA NORTE", "CLN 215", "CLN 200"),
        ("ASA NORTE", "SHCGN 703", "SHCGN 700"),
        # prefixo fora da lista do bairro não vira grupo
        ("ASA NORTE", "SEPN 515", ""),
        # sem série de centena
        ("ASA NORTE", "SHN QUADRA 1", ""),
        ("ASA NORTE", "", ""),
        ("ASA SUL", "SQS 116", "SQS 100"),
        ("ASA SUL", "SQS 403 BLOCO C", "SQS 400"),
        ("ASA SUL", "SEPS 705/905", "SEPS 700"),
        ("ASA SUL", "SRTVS CONJUNTO L", ""),
        # a regra dos lagos continua valendo
        ("LAGO SUL", "QI 5 CONJUNTO 3", "QI - Início"),
        # bairro sem regra mantém a quadra crua
        ("GUARA", "QE 40 CONJUNTO A", "QE 40 CONJUNTO A"),
    ]
    for bairro, bruta, esperado in casos:
        obtido = em._mapear_quadra(bruta, bairro)
        assert obtido == esperado, f"{bairro} / {bruta!r}: {obtido!r} != {esperado!r}"
    print(f"OK  agrupamento de quadras ({len(casos)} casos)")


def teste_templates_sql():
    """
    `psql.SQL(...).format()` usa string.Formatter: qualquer `{...}` no template
    vira placeholder. Um regex como `[0-9]{4,}` explode em
    `KeyError: '4,'` — só na hora de rodar contra o banco, que é onde não se
    quer descobrir. Aqui a checagem é estática: todo campo entre chaves tem
    que ser um nome que o `.format()` daquele arquivo realmente fornece.
    """
    import re
    from string import Formatter

    for arquivo in ("acionador.py", "listagem_historico.py", "diagnostico_portais.py"):
        caminho = os.path.join(os.path.dirname(os.path.abspath(__file__)), arquivo)
        fonte = open(caminho, encoding="utf-8").read()
        for bloco in re.findall(r'psql\.SQL\("""(.*?)"""\)', fonte, re.S):
            campos = {nome for _, nome, _, _ in Formatter().parse(bloco) if nome}
            suspeitos = {c for c in campos if not c.isidentifier()}
            assert not suspeitos, (
                f"{arquivo}: chave literal não escapada em template SQL: "
                f"{sorted(suspeitos)} — dobre as chaves (ex.: {{{{4,}}}})"
            )
    print("OK  templates SQL (chaves literais escapadas)")


def teste_cluster_por_faixa():
    """
    Lagos: cada faixa de luxo com os 3 rótulos, sem grupo residual e sem
    sobreposição de m² entre rótulos da mesma faixa. Reproduz o caso real do
    LAGO SUL: ALTO LUXO concentrado em m² alto com cauda longa, que no
    cluster do bairro inteiro ficava sem "Original".
    """
    rng = np.random.default_rng(7)
    em = EstudoMercado(bairro="LAGO SUL", tipo="CASA CONDOMINIO")

    def faixa(n, preco_min, preco_max, m2_med, cauda=0):
        m2 = rng.lognormal(np.log(m2_med), 0.18, n)
        m2[:cauda] = rng.uniform(32_000, 38_000, cauda)   # cauda de outliers
        preco = rng.uniform(preco_min, preco_max, n)
        return pd.DataFrame({"valor_m2": m2, "preco": preco, "area_util": preco / m2})

    dados = pd.concat([
        faixa(180, 8_100_000, 19_000_000, 19_000, cauda=8),   # ALTO LUXO
        faixa(240, 2_500_000, 7_900_000, 8_500, cauda=5),     # LUXO
        faixa(12, 600_000, 2_400_000, 4_000),                 # abaixo: pequeno
    ], ignore_index=True)

    cortes = em._treinar_cluster_por_faixa(dados, "teste")
    assert set(cortes) == {"ALTO LUXO", "LUXO"}, f"faixas com cluster: {set(cortes)}"

    out = em._aplicar_cluster_por_faixa(dados, cortes, "LAGO SUL")
    assert (out["luxo"] != "").all(), "faixa sem amostra não pode entrar no cluster"
    for f, g in out.groupby("luxo"):
        parc = g["cluster_nome"].value_counts(normalize=True)
        assert set(parc.index) == {"Original", "Reformado", "Nova"}, f"{f}: {dict(parc)}"
        assert parc.min() >= em.CLUSTER_FAIXA_MIN_PARCELA - 0.01, f"{f}: grupo residual {dict(parc)}"
        faixas_m2 = g.groupby("cluster_nome")["valor_m2"].agg(["min", "max"])
        assert faixas_m2.loc["Original", "max"] < faixas_m2.loc["Reformado", "min"], f"{f}: sobreposição"
        assert faixas_m2.loc["Reformado", "max"] < faixas_m2.loc["Nova", "min"], f"{f}: sobreposição"

    # Treinado no período todo e aplicado a um pedaço: o rótulo não muda.
    parte = em._aplicar_cluster_por_faixa(dados.iloc[::3], cortes, "LAGO SUL")
    assert (parte["cluster_nome"] == out.loc[parte.index, "cluster_nome"]).all()
    print("OK  cluster por faixa de luxo (3 rótulos por faixa, sem mistura)")


def teste_coerencia_amostra_baixa():
    """
    Amostra baixa fica quando o valor é coerente com a série ou com o pai, e
    sai quando diverge. Casos tirados do LAGO SUL real: casa < 75 m² a
    R$ 37.282/m² com 1 imóvel (área errada no anúncio) contra pai de 8.249.
    """
    em = EstudoMercado(bairro="LAGO SUL", tipo="CASA")

    def ponto(seg, mes, n, m2, **chave):
        base = dict(segmento=seg, mes_ref=mes, vaga_cat="COM VAGA", luxo="LUXO",
                    cluster_nome="", quartos=-1, metragem_fx="", quadra="",
                    amostra=n, imoveis_unicos=n, m2_mediana=m2)
        base.update(chave)
        return base

    out = pd.DataFrame([
        ponto("GERAL_VAGA", "2026-10", 120, 8_249),
        ponto("GERAL_VAGA", "2026-09", 110, 8_100),
        # METRAGEM <75: 1 imóvel a 37 mil/m² — lixo, sai
        ponto("METRAGEM_VAGA", "2026-10", 1, 37_282, metragem_fx="<75"),
        # METRAGEM 1500-2000: 3 imóveis, 40% abaixo do pai — casa grande, fica
        ponto("METRAGEM_VAGA", "2026-10", 3, 4_950, metragem_fx="1500-2000"),
        # QUARTOS 5: forte em set, fraco em out e coerente com a série — fica
        ponto("QUARTOS_VAGA", "2026-09", 20, 7_900, quartos=5),
        ponto("QUARTOS_VAGA", "2026-10", 4, 8_300, quartos=5),
        # QUARTOS 1: forte em set, fraco em out com valor de piso — sai
        ponto("QUARTOS_VAGA", "2026-09", 18, 7_000, quartos=1),
        ponto("QUARTOS_VAGA", "2026-10", 2, 2_400, quartos=1),
        # ALTO LUXO sem pai robusto nem série robusta: corte antigo (>= 5)
        ponto("QUADRA_VAGA", "2026-10", 6, 21_000, quadra="QI - Início", luxo="ALTO LUXO"),
        ponto("QUADRA_VAGA", "2026-10", 2, 19_000, quadra="QI - Final", luxo="ALTO LUXO"),
    ])
    out["confiabilidade"] = em._classificar_confianca(out["imoveis_unicos"])
    r = em._conferir_coerencia(out).set_index(["segmento", "mes_ref", "metragem_fx", "quartos", "quadra"])

    def tem(*k):
        return k in r.index

    assert not tem("METRAGEM_VAGA", "2026-10", "<75", -1, ""), "lixo de 37 mil/m² passou"
    assert tem("METRAGEM_VAGA", "2026-10", "1500-2000", -1, ""), "casa grande coerente caiu"
    assert r.loc[("METRAGEM_VAGA", "2026-10", "1500-2000", -1, ""), "coerencia"] == "PAI"
    assert tem("QUARTOS_VAGA", "2026-10", "", 5, ""), "ponto fraco coerente com a série caiu"
    assert r.loc[("QUARTOS_VAGA", "2026-10", "", 5, ""), "coerencia"] == "SERIE"
    assert not tem("QUARTOS_VAGA", "2026-10", "", 1, ""), "ponto no piso passou"
    assert tem("QUARTOS_VAGA", "2026-09", "", 1, ""), "mês robusto da série não pode cair"
    assert tem("QUADRA_VAGA", "2026-10", "", -1, "QI - Início"), "sem referência e n>=5 deve ficar"
    assert not tem("QUADRA_VAGA", "2026-10", "", -1, "QI - Final"), "sem referência e n<5 deve sair"
    assert (r.loc[r["confiabilidade"].isin(["ALTA", "MEDIA"]), "coerencia"] == "ROBUSTA").all()
    print("OK  amostra baixa: coerente fica, divergente sai")


if __name__ == "__main__":
    df_raw = _base_sintetica()
    print(f"base sintética: {len(df_raw)} linhas, {df_raw['codigo'].nunique()} códigos\n")
    teste_limpeza(df_raw)
    teste_colunas_e_percentis(df_raw)
    teste_indice_repeat(df_raw)
    teste_cauda(df_raw)
    teste_payload_upsert(df_raw)
    teste_agrupamento_quadras()
    teste_cluster_por_faixa()
    teste_coerencia_amostra_baixa()
    teste_templates_sql()
    print("\nTODOS OS TESTES PASSARAM")
