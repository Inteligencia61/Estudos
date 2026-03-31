from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2 import sql as psql


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # fonte (raw)
    "TABELA": "imoveis",

    # filtros do estudo (default)
    "BAIRRO_ALVO": "ASA SUL",
    "TIPO_ALVO": "CASA",
    "OFERTA_ALVO": "VENDA",

    "PRECO_MIN": 500_000,
    "PRECO_MAX": 50_000_000,
    "AREA_MIN": 40,
    "AREA_MAX": 1_500_000,

    "VLM2_MIN": 1_000,
    "VLM2_MAX": 900_000,

    "APLICAR_IQR": True,

    # clustering padrão
    "CLUSTERS_ATIVOS": True,
    "MIN_AMOSTRA_PARA_CLUSTER": 10,
    "KMEANS_N_CLUSTERS": 9,
    "RANDOM_STATE": 42,

    # cluster luxo x super luxo
    "BAIRROS_LUXO": ["LAGO SUL", "LAGO NORTE"],
    "MIN_AMOSTRA_PARA_CLUSTER_LUXO": 12,

    # meses-alvo
    "MESES_ALVO": ["2025-12", "2026-01", "2026-02"],

    # cluster fixo: intervalo de treino extra
    "CLUSTER_TREINO_EXTRA_MESES_ANTES": 0,
    "CLUSTER_TREINO_EXTRA_MESES_DEPOIS": 0,

    # destino
    "SCHEMA_ANALYTICS": "analytics",
    "TBL_METRICAS": "estudo_metricas",

    # performance
    "UPSERT_PAGE_SIZE": 2000,

    # regras de segmentos
    "MIN_AMOSTRA_SEGMENTO": 3,

    # faixas de metragem
    "METRAGEM_BINS": [0, 75, 90, 130, 160, 200, 400, 600, 800, 1000, 10_000_000],
    "METRAGEM_LABELS": ["<75", "75-90", "90-130", "130-160", "160-200", "200-400", "400-600", "600-800", "800-1000", ">1000"],
}


# ============================================================
# Helpers de data
# ============================================================

def first_day_of_month(ym: str) -> date:
    y, m = map(int, ym.split("-"))
    return date(y, m, 1)


def last_day_of_month(ym: str) -> date:
    d1 = first_day_of_month(ym)
    if d1.month == 12:
        d2 = date(d1.year + 1, 1, 1)
    else:
        d2 = date(d1.year, d1.month + 1, 1)
    return (pd.Timestamp(d2) - pd.Timedelta(days=1)).date()


def add_months(d: date, n: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=n)).date()


def janela_3_meses_para_mes_alvo(ym: str) -> Tuple[date, date]:
    fim = last_day_of_month(ym)
    inicio_mes_alvo = first_day_of_month(ym)
    inicio = add_months(inicio_mes_alvo, -2)
    return inicio, fim


# ============================================================
# Postgres
# ============================================================

def pg_connect():
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    db = os.getenv("PGDATABASE", "postgres")
    user = os.getenv("PGUSER", "postgres")
    pwd = os.getenv("PGPASSWORD", "")
    return psycopg2.connect(host=host, port=port, dbname=db, user=user, password=pwd)


def ensure_schema_and_table(conn) -> None:
    schema = CONFIG["SCHEMA_ANALYTICS"]
    tbl = CONFIG["TBL_METRICAS"]
    raw = CONFIG["TABELA"]

    ddl = f"""
    create schema if not exists {schema};

    create table if not exists {schema}.{tbl} (
      bairro         text not null,
      tipo           text not null,
      oferta         text not null,
      mes_alvo       text not null,
      janela_inicio  date not null,
      janela_fim     date not null,
      mes_ref        text not null,
      segmento       text not null,
      vaga_cat       text not null,
      cluster_nome   text not null default '',
      quartos        int  not null default -1,
      metragem_fx    text not null default '',
      quadra         text not null default '',
      luxo           text not null default '',
      amostra        int not null,
      m2_medio       double precision,
      m2_mediana     double precision,
      preco_mediana  double precision,
      area_mediana   double precision,
      gerado_em      timestamp not null default now(),
      primary key (
        bairro, tipo, oferta,
        mes_alvo, janela_inicio, janela_fim,
        mes_ref, segmento,
        vaga_cat, cluster_nome, quartos, metragem_fx, quadra, luxo
      )
    );

    create index if not exists idx_{tbl}_filtros
      on {schema}.{tbl} (bairro, tipo, oferta, mes_alvo, segmento, mes_ref);

    create index if not exists idx_{raw}_data_coleta on {raw} (data_coleta);
    create index if not exists idx_{raw}_filtros_data on {raw} (bairro, tipo, oferta, data_coleta);
    create index if not exists idx_{raw}_codigo_data on {raw} (codigo, data_coleta);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)

    with conn.cursor() as cur:
        cur.execute(f"""
            alter table {schema}.{tbl}
            add column if not exists luxo text not null default '';
        """)


def carregar_do_banco_filtrado(inicio: date, fim: date) -> pd.DataFrame:
    tabela = CONFIG["TABELA"]
    bairro = CONFIG["BAIRRO_ALVO"]
    tipo = CONFIG["TIPO_ALVO"]

    oferta_alvo = CONFIG["OFERTA_ALVO"]
    ofertas_aceitas = list({oferta_alvo, "PUBLICADO", "VENDA"})

    preco_min, preco_max = CONFIG["PRECO_MIN"], CONFIG["PRECO_MAX"]
    area_min, area_max = CONFIG["AREA_MIN"], CONFIG["AREA_MAX"]

    q = psql.SQL("""
        SELECT DISTINCT ON (codigo, data_coleta)
            TRIM(codigo)::text                      as codigo,
            UPPER(TRIM(bairro))::text               as bairro,
            UPPER(TRIM(cidade))::text               as cidade,
            UPPER(TRIM(tipo))::text                 as tipo,
            UPPER(TRIM(oferta))::text               as oferta,
            area_util::double precision             as area_util,
            preco::double precision                 as preco,
            quartos::double precision               as quartos,
            vagas::double precision                 as vagas,
            latitude::double precision              as latitude,
            longitude::double precision             as longitude,
            UPPER(TRIM(quadra))::text               as quadra,
            data_coleta::date                       as data_coleta
        FROM {tabela}
        WHERE data_coleta >= %s
          AND data_coleta <= %s
          AND UPPER(TRIM(bairro)) = %s
          AND UPPER(TRIM(tipo)) = %s
          AND UPPER(TRIM(oferta)) = ANY(%s)
          AND preco is not null
          AND area_util is not null
          AND preco >= %s AND preco <= %s
          AND area_util >= %s AND area_util <= %s
        ORDER BY codigo, data_coleta, data_coleta DESC;
    """).format(tabela=psql.Identifier(tabela))

    with pg_connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                q,
                (inicio, fim, bairro, tipo, ofertas_aceitas, preco_min, preco_max, area_min, area_max)
            )
            rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce").dt.date

    for c in ["preco", "area_util", "quartos", "vagas", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["bairro", "cidade", "tipo", "oferta", "quadra", "codigo"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.upper()

    return df


# ============================================================
# Limpeza / derivações
# ============================================================

def remover_outliers_iqr(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    if coluna not in df.columns or df.empty:
        return df

    s = df[coluna].dropna()
    if s.empty:
        return df

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return df

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return df[(df[coluna] >= low) & (df[coluna] <= high)].copy()


def limpar_dados_db(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df = df.dropna(subset=["preco", "area_util"])
    df["valor_m2"] = df["preco"] / df["area_util"]

    df = df[(df["valor_m2"] >= CONFIG["VLM2_MIN"]) & (df["valor_m2"] <= CONFIG["VLM2_MAX"])]

    if CONFIG["APLICAR_IQR"] and len(df) >= 20:
        df = remover_outliers_iqr(df, "valor_m2")

    df["quartos"] = pd.to_numeric(df["quartos"], errors="coerce").fillna(0).astype(int)
    df["vagas"] = pd.to_numeric(df["vagas"], errors="coerce").fillna(0).astype(int)

    df["data_dt"] = pd.to_datetime(df["data_coleta"], errors="coerce")
    df["mes_ref"] = df["data_dt"].dt.to_period("M").astype("string")
    df["vaga_cat"] = np.where(df["vagas"] > 0, "COM VAGA", "SEM VAGA")

    df["metragem_fx"] = pd.cut(
        df["area_util"],
        bins=CONFIG["METRAGEM_BINS"],
        labels=CONFIG["METRAGEM_LABELS"],
        include_lowest=True,
        right=False
    ).astype("string").fillna("")

    if "quadra" in df.columns:
        df["quadra"] = df["quadra"].astype("string").fillna("").str.strip().str.upper()
    else:
        df["quadra"] = ""

    # coluna luxo começa vazia; será preenchida pela clusterização de luxo
    df["luxo"] = ""

    return df


# ============================================================
# Cluster fixo padrão
# ============================================================

def treinar_modelo_cluster_global(df_treino: pd.DataFrame) -> Tuple[StandardScaler, KMeans, Dict[int, str]]:
    if not SKLEARN_OK:
        raise RuntimeError("sklearn indisponível no ambiente.")
    if df_treino.empty or len(df_treino) < CONFIG["MIN_AMOSTRA_PARA_CLUSTER"]:
        raise RuntimeError("Amostra insuficiente para treinar cluster global.")

    feats = ["valor_m2", "area_util"]
    base = df_treino.dropna(subset=feats).copy()
    if len(base) < CONFIG["MIN_AMOSTRA_PARA_CLUSTER"]:
        raise RuntimeError("Amostra insuficiente (após dropna) para treinar cluster global.")

    scaler = StandardScaler()
    X = scaler.fit_transform(base[feats])

    km = KMeans(
        n_clusters=CONFIG["KMEANS_N_CLUSTERS"],
        random_state=CONFIG["RANDOM_STATE"],
        n_init=10
    )
    km.fit(X)

    centers_real = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers_real, columns=feats)

    order = centers_df.sort_values("valor_m2").index.tolist()
    labels = ["01 - Original", "02 - Semi-Reformado", "03 - Reformado"]
    mapping = {cid: labels[min(i // 3, 2)] for i, cid in enumerate(order)}
    return scaler, km, mapping


def aplicar_cluster_fixo(dados: pd.DataFrame, scaler: StandardScaler, km: KMeans, mapping: Dict[int, str]) -> pd.DataFrame:
    feats = ["valor_m2", "area_util"]
    dfc = dados.dropna(subset=feats).copy()
    if dfc.empty:
        return dfc

    X = scaler.transform(dfc[feats])
    dfc["cluster_id"] = km.predict(X)
    dfc["cluster_nome"] = dfc["cluster_id"].map(mapping).astype("string").fillna("")
    return dfc


# ============================================================
# Cluster luxo x super luxo
# ============================================================

def bairro_permite_cluster_luxo(bairro: str) -> bool:
    bairro_norm = str(bairro).strip().upper()
    return bairro_norm in {b.upper() for b in CONFIG["BAIRROS_LUXO"]}


def aplicar_cluster_luxo_super_luxo(dados: pd.DataFrame) -> pd.DataFrame:
    """
    Só segmenta quando o bairro for LAGO SUL ou LAGO NORTE.
    Faz KMeans com 2 grupos usando:
      - preco
      - valor_m2
      - area_util

    Cluster de menor preço médio => LUXO
    Cluster de maior preço médio => SUPER LUXO
    """
    df = dados.copy()
    if df.empty:
        df["luxo"] = ""
        return df

    bairro_ref = str(df["bairro"].dropna().astype(str).iloc[0]).strip().upper() if "bairro" in df.columns and df["bairro"].notna().any() else ""
    if not bairro_permite_cluster_luxo(bairro_ref):
        df["luxo"] = ""
        return df

    if not SKLEARN_OK:
        df["luxo"] = ""
        return df

    colunas = ["preco", "valor_m2", "area_util"]
    base = df.dropna(subset=colunas).copy()

    min_amostra = int(CONFIG["MIN_AMOSTRA_PARA_CLUSTER_LUXO"])
    if len(base) < min_amostra:
        df["luxo"] = ""
        return df

    scaler = StandardScaler()
    X = scaler.fit_transform(base[colunas])

    km = KMeans(n_clusters=2, random_state=CONFIG["RANDOM_STATE"], n_init=10)
    base["cluster_luxo_id"] = km.fit_predict(X)

    order = (
        base.groupby("cluster_luxo_id")["preco"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    mapping = {
        order[0]: "LUXO",
        order[1]: "SUPER LUXO",
    }

    base["luxo"] = base["cluster_luxo_id"].map(mapping).astype("string").fillna("")

    # traz de volta para o df original
    df["luxo"] = ""
    df.loc[base.index, "luxo"] = base["luxo"]

    return df


# ============================================================
# Agregação
# ============================================================

def _agg_metricas(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            amostra=("valor_m2", "size"),
            m2_medio=("valor_m2", "mean"),
            m2_mediana=("valor_m2", "median"),
            preco_mediana=("preco", "median"),
            area_mediana=("area_util", "median"),
        )
        .reset_index()
    )
    return out


def build_metricas_long(
    dados_janela: pd.DataFrame,
    mes_alvo: str,
    inicio: date,
    fim: date,
    scaler: Optional[StandardScaler],
    km: Optional[KMeans],
    mapping: Optional[Dict[int, str]],
) -> pd.DataFrame:
    """
    Gera linhas long para:
      - GERAL_VAGA
      - CLUSTER_VAGA
      - QUARTOS_VAGA
      - METRAGEM_VAGA
      - QUADRA_VAGA

    E aplica LUXO x SUPER LUXO somente se o bairro permitir.
    """
    if dados_janela.empty:
        return pd.DataFrame()

    base = dados_janela.copy()
    base = base.dropna(subset=["mes_ref", "vaga_cat", "valor_m2", "preco", "area_util"])
    if base.empty:
        return pd.DataFrame()

    # aplica cluster de luxo/super luxo na janela inteira
    base = aplicar_cluster_luxo_super_luxo(base)
    if "luxo" not in base.columns:
        base["luxo"] = ""
    base["luxo"] = base["luxo"].astype("string").fillna("")

    linhas = []

    # 1) geral por vaga e luxo
    g = _agg_metricas(base, ["mes_ref", "vaga_cat", "luxo"])
    if not g.empty:
        g["segmento"] = "GERAL_VAGA"
        g["cluster_nome"] = ""
        g["quartos"] = -1
        g["metragem_fx"] = ""
        g["quadra"] = ""
        linhas.append(g)

    # 2) cluster padrão por vaga e luxo
    if CONFIG["CLUSTERS_ATIVOS"] and scaler is not None and km is not None and mapping is not None:
        dfc = aplicar_cluster_fixo(base, scaler, km, mapping)
        if not dfc.empty:
            if "luxo" not in dfc.columns:
                dfc["luxo"] = ""
            dfc["luxo"] = dfc["luxo"].astype("string").fillna("")

            c = _agg_metricas(dfc, ["mes_ref", "vaga_cat", "cluster_nome", "luxo"])
            if not c.empty:
                c["segmento"] = "CLUSTER_VAGA"
                c["quartos"] = -1
                c["metragem_fx"] = ""
                c["quadra"] = ""
                linhas.append(c)

    # 3) quartos por vaga e luxo
    qbase = base[base["quartos"] > 0].copy()
    if not qbase.empty:
        q = _agg_metricas(qbase, ["mes_ref", "vaga_cat", "quartos", "luxo"])
        if not q.empty:
            q["segmento"] = "QUARTOS_VAGA"
            q["cluster_nome"] = ""
            q["metragem_fx"] = ""
            q["quadra"] = ""
            linhas.append(q)

    # 4) metragem por vaga e luxo
    mbase = base[base["metragem_fx"].astype(str).str.len() > 0].copy()
    if not mbase.empty:
        m = _agg_metricas(mbase, ["mes_ref", "vaga_cat", "metragem_fx", "luxo"])
        if not m.empty:
            m["segmento"] = "METRAGEM_VAGA"
            m["cluster_nome"] = ""
            m["quartos"] = -1
            m["quadra"] = ""
            linhas.append(m)

    # 5) quadra por vaga e luxo
    qd = base[base["quadra"].astype(str).str.len() > 0].copy()
    if not qd.empty:
        qq = _agg_metricas(qd, ["mes_ref", "vaga_cat", "quadra", "luxo"])
        if not qq.empty:
            qq["segmento"] = "QUADRA_VAGA"
            qq["cluster_nome"] = ""
            qq["quartos"] = -1
            qq["metragem_fx"] = ""
            linhas.append(qq)

    if not linhas:
        return pd.DataFrame()

    out = pd.concat(linhas, ignore_index=True)

    out = out[out["amostra"] >= int(CONFIG["MIN_AMOSTRA_SEGMENTO"])].copy()
    if out.empty:
        return out

    out["bairro"] = CONFIG["BAIRRO_ALVO"]
    out["tipo"] = CONFIG["TIPO_ALVO"]
    out["oferta"] = CONFIG["OFERTA_ALVO"]
    out["mes_alvo"] = mes_alvo
    out["janela_inicio"] = inicio
    out["janela_fim"] = fim

    out["cluster_nome"] = out.get("cluster_nome", "").astype("string").fillna("")
    out["metragem_fx"] = out.get("metragem_fx", "").astype("string").fillna("")
    out["quadra"] = out.get("quadra", "").astype("string").fillna("")
    out["luxo"] = out.get("luxo", "").astype("string").fillna("")
    out["quartos"] = pd.to_numeric(out.get("quartos", -1), errors="coerce").fillna(-1).astype(int)

    return out[
        [
            "bairro", "tipo", "oferta",
            "mes_alvo", "janela_inicio", "janela_fim",
            "mes_ref", "segmento",
            "vaga_cat", "cluster_nome", "quartos", "metragem_fx", "quadra", "luxo",
            "amostra", "m2_medio", "m2_mediana", "preco_mediana", "area_mediana",
        ]
    ]


# ============================================================
# UPSERT
# ============================================================

def upsert_metricas_long(conn, df_long: pd.DataFrame) -> None:
    if df_long.empty:
        return

    schema = CONFIG["SCHEMA_ANALYTICS"]
    tbl = CONFIG["TBL_METRICAS"]

    cols = [
        "bairro", "tipo", "oferta",
        "mes_alvo", "janela_inicio", "janela_fim",
        "mes_ref", "segmento",
        "vaga_cat", "cluster_nome", "quartos", "metragem_fx", "quadra", "luxo",
        "amostra", "m2_medio", "m2_mediana", "preco_mediana", "area_mediana",
    ]

    payload = []
    for _, r in df_long.iterrows():
        payload.append(tuple(r[c] if pd.notna(r[c]) else None for c in cols))

    q = psql.SQL(f"""
        insert into {{schema}}.{{tbl}} (
          bairro, tipo, oferta,
          mes_alvo, janela_inicio, janela_fim,
          mes_ref, segmento,
          vaga_cat, cluster_nome, quartos, metragem_fx, quadra, luxo,
          amostra, m2_medio, m2_mediana, preco_mediana, area_mediana
        ) values %s
    """).format(schema=psql.Identifier(schema), tbl=psql.Identifier(tbl))

    with conn.cursor() as cur:
        execute_values(cur, q.as_string(conn), payload, page_size=int(CONFIG["UPSERT_PAGE_SIZE"]))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gera métricas long por janela de 3 meses e grava em analytics.estudo_metricas."
    )
    parser.add_argument("--meses", nargs="*", default=CONFIG["MESES_ALVO"],
                        help="Meses-alvo YYYY-MM. Ex: --meses 2025-11 2025-12 2026-01")
    parser.add_argument("--bairro", default=CONFIG["BAIRRO_ALVO"], help="Bairro. Ex: ASA SUL")
    parser.add_argument("--tipo", default=CONFIG["TIPO_ALVO"], help="Tipo. Ex: APARTAMENTO")
    parser.add_argument("--oferta", default=CONFIG["OFERTA_ALVO"], help="Oferta. Ex: VENDA")
    parser.add_argument("--sem-cluster", action="store_true", help="Desliga clustering padrão.")
    args = parser.parse_args()

    CONFIG["BAIRRO_ALVO"] = str(args.bairro).strip().upper()
    CONFIG["TIPO_ALVO"] = str(args.tipo).strip().upper()
    CONFIG["OFERTA_ALVO"] = str(args.oferta).strip().upper()

    if args.sem_cluster:
        CONFIG["CLUSTERS_ATIVOS"] = False

    meses_alvo = args.meses

    janelas = []
    for ym in meses_alvo:
        ini, fim = janela_3_meses_para_mes_alvo(ym)
        janelas.append((ym, ini, fim))

    inicio_global = min(x[1] for x in janelas)
    fim_global = max(x[2] for x in janelas)

    treino_ini = add_months(inicio_global, -int(CONFIG["CLUSTER_TREINO_EXTRA_MESES_ANTES"]))
    treino_fim = add_months(fim_global, int(CONFIG["CLUSTER_TREINO_EXTRA_MESES_DEPOIS"]))

    df = carregar_do_banco_filtrado(treino_ini, treino_fim)
    if df.empty:
        print("Nenhum dado encontrado para os filtros informados.")
        return

    df_limpo = limpar_dados_db(df)
    if df_limpo.empty:
        print("Nenhum dado válido após limpeza.")
        return

    scaler = km = mapping = None
    if CONFIG["CLUSTERS_ATIVOS"]:
        try:
            scaler, km, mapping = treinar_modelo_cluster_global(df_limpo)
        except Exception as e:
            print(f"[WARN] Cluster padrão desativado por erro no treino: {e}")
            scaler = km = mapping = None

    with pg_connect() as conn:
        ensure_schema_and_table(conn)

        for (ym, ini, fim) in janelas:
            dados_janela = df_limpo[
                (df_limpo["data_coleta"] >= ini) & (df_limpo["data_coleta"] <= fim)
            ].copy()

            if dados_janela.empty:
                print(f"[INFO] Sem dados para a janela do mês-alvo {ym}.")
                continue

            df_long = build_metricas_long(
                dados_janela=dados_janela,
                mes_alvo=ym,
                inicio=ini,
                fim=fim,
                scaler=scaler,
                km=km,
                mapping=mapping,
            )

            if df_long.empty:
                print(f"[INFO] Sem métricas para gravar no mês-alvo {ym}.")
                continue

            upsert_metricas_long(conn, df_long)
            print(f"[OK] Métricas gravadas para {ym}: {len(df_long)} linhas.")

        conn.commit()

    print("Processamento concluído com sucesso.")


if __name__ == "__main__":
    main()