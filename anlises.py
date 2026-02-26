from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

import psycopg2
from psycopg2.extras import RealDictCursor

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "TABELA": "imoveis",  # ajuste se necessário

    "BAIRRO_ALVO": "ASA SUL",
    "TIPO_ALVO": "APARTAMENTO",
    "OFERTA_ALVO": "VENDA",

    "PRECO_MIN": 1_000_000,
    "PRECO_MAX": 30_000_000,
    "AREA_MIN": 40,
    "AREA_MAX": 50_000,

    "VLM2_MIN": 6_000,
    "VLM2_MAX": 900_000,

    "APLICAR_IQR": True,

    "ANALISAR_POR_QUARTOS": True,
    "ANALISAR_POR_VAGA": True,

    "CLUSTERS_ATIVOS": True,
    "MIN_AMOSTRA_PARA_CLUSTER": 20,

    # kmeans “base” (9 -> 3 faixas)
    "KMEANS_N_CLUSTERS": 9,
    "RANDOM_STATE": 42,

    # meses-alvo (padrão)
    "MESES_ALVO": ["2025-11", "2025-12", "2026-01"],

    # ===== NOVO: cluster FIXO =====
    # Por padrão, o modelo é treinado no intervalo global que cobre todas as janelas
    # (ex.: 2025-09-01 -> 2026-01-31). Você pode expandir isso se quiser.
    "CLUSTER_TREINO_EXTRA_MESES_ANTES": 0,  # ex.: 3 para treinar com +3 meses antes do início_global
    "CLUSTER_TREINO_EXTRA_MESES_DEPOIS": 0,  # ex.: 1 para treinar com +1 mês após fim_global
}

# ============================================================
# Helpers de data (janelas mensais)
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
    ts = pd.Timestamp(d) + pd.DateOffset(months=n)
    return ts.date()

def janela_3_meses_para_mes_alvo(ym: str) -> Tuple[date, date]:
    """
    Retorna (inicio, fim) da janela de 3 meses INCLUSIVA do mês-alvo.
    Ex: 2026-01 -> 2025-11-01 até 2026-01-31
    """
    fim = last_day_of_month(ym)
    inicio_mes_alvo = first_day_of_month(ym)
    inicio = add_months(inicio_mes_alvo, -2)
    return inicio, fim

# ============================================================
# Conexão e Query Postgres
# ============================================================

def pg_connect():
    host = os.getenv("PGHOST", "localhost")
    port = int(os.getenv("PGPORT", "5432"))
    db = os.getenv("PGDATABASE", "postgres")
    user = os.getenv("PGUSER", "postgres")
    pwd = os.getenv("PGPASSWORD", "")
    return psycopg2.connect(host=host, port=port, dbname=db, user=user, password=pwd)

def carregar_do_banco(inicio: date, fim: date) -> pd.DataFrame:
    tabela = CONFIG["TABELA"]

    sql = f"""
        SELECT
            codigo,
            bairro,
            cidade,
            tipo,
            oferta,
            area_util,
            preco,
            quartos,
            vagas,
            latitude,
            longitude,
            quadra,
            data_coleta
        FROM {tabela}
        WHERE data_coleta >= %s
          AND data_coleta <= %s;
    """

    with pg_connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (inicio, fim))
            rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce").dt.date
    for c in ["preco", "area_util", "quartos", "vagas", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["bairro", "cidade", "tipo", "oferta", "quadra"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.upper()

    df["codigo"] = df["codigo"].astype("string").str.strip()
    return df

# ============================================================
# Limpeza (padrão do banco)
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

    if "codigo" in df.columns and "data_coleta" in df.columns:
        df = df.drop_duplicates(subset=["codigo", "data_coleta"], keep="last")

    df = df[df["bairro"] == CONFIG["BAIRRO_ALVO"]]
    df = df[df["tipo"] == CONFIG["TIPO_ALVO"]]
    df = df[df["oferta"].isin([CONFIG["OFERTA_ALVO"], "PUBLICADO", "VENDA"])]

    df = df.dropna(subset=["preco", "area_util"])
    df = df[(df["preco"] >= CONFIG["PRECO_MIN"]) & (df["preco"] <= CONFIG["PRECO_MAX"])]
    df = df[(df["area_util"] >= CONFIG["AREA_MIN"]) & (df["area_util"] <= CONFIG["AREA_MAX"])]

    df["valor_m2"] = df["preco"] / df["area_util"]

    df = df[(df["valor_m2"] >= CONFIG["VLM2_MIN"]) & (df["valor_m2"] <= CONFIG["VLM2_MAX"])]

    if CONFIG["APLICAR_IQR"] and len(df) >= 20:
        df = remover_outliers_iqr(df, "valor_m2")

    df["quartos"] = df["quartos"].fillna(0).astype(int)
    df["vagas"] = df["vagas"].fillna(0).astype(int)

    df["data_dt"] = pd.to_datetime(df["data_coleta"], errors="coerce")
    df["mes"] = df["data_dt"].dt.to_period("M").astype("string")
    return df

# ============================================================
# Resumos
# ============================================================

def resumo_estatistico(df: pd.DataFrame, label: str) -> Dict[str, object]:
    if df.empty:
        return {"segmento": label, "amostra": 0}

    vl = df["valor_m2"].dropna()
    pr = df["preco"].dropna()

    return {
        "segmento": label,
        "amostra": int(len(df)),
        "m2_medio": float(vl.mean()),
        "m2_mediana": float(vl.median()),
        "m2_p10": float(vl.quantile(0.10)),
        "m2_p90": float(vl.quantile(0.90)),
        "preco_medio": float(pr.mean()),
        "preco_mediana": float(pr.median()),
        "area_media": float(df["area_util"].mean()),
        "area_mediana": float(df["area_util"].median()),
    }

def imprimir_tabela(titulo: str, rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)
    if not rows:
        print("Sem dados.")
        return

    out = pd.DataFrame(rows)

    for c in ["m2_medio", "m2_mediana", "m2_p10", "m2_p90", "preco_medio", "preco_mediana"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: "-" if pd.isna(x) else f"{x:,.0f}".replace(",", "."))

    for c in ["area_media", "area_mediana"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: "-" if pd.isna(x) else f"{x:,.1f}".replace(",", "."))

    print(out.to_string(index=False))

# ============================================================
# Gráfico: valores por mês (dentro da janela)
# ============================================================

def plot_valores_por_mes(dados: pd.DataFrame, out_prefix: str) -> pd.DataFrame:
    base = dados.dropna(subset=["data_dt"]).copy()
    if base.empty:
        print("[INFO] Sem datas válidas para gráfico mensal.")
        return pd.DataFrame()

    agg_mes = (
        base.groupby("mes", dropna=False)
        .agg(
            amostra=("mes", "size"),
            m2_mediana=("valor_m2", "median"),
            m2_medio=("valor_m2", "mean"),
            preco_mediana=("preco", "median"),
            area_mediana=("area_util", "median"),
        )
        .reset_index()
        .sort_values("mes")
    )

    plt.figure(figsize=(12, 6))
    plt.plot(agg_mes["mes"], agg_mes["m2_mediana"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Valor/m² (mediana) por mês - {CONFIG['BAIRRO_ALVO']} | janela {out_prefix}")
    plt.xlabel("Mês (YYYY-MM)")
    plt.ylabel("R$/m² (mediana)")
    for i, row in agg_mes.reset_index(drop=True).iterrows():
        plt.text(i, row["m2_mediana"], f" n={int(row['amostra'])}", fontsize=9, va="bottom", ha="center")
    plt.tight_layout()
    fname = f"{out_prefix}_valores_por_mes.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico salvo: {fname}")

    return agg_mes

# ============================================================
# ===== NOVO: CLUSTER FIXO (treina UMA vez no período global)
# ============================================================

def treinar_modelo_cluster_global(df_treino: pd.DataFrame) -> Tuple[StandardScaler, KMeans, Dict[int, str]]:
    """
    Treina scaler + kmeans UMA vez com base global e gera mapping fixo:
    cluster_id -> "01/02/03".
    Esse mapping é baseado nos centroides (valor_m2), para ficar estável no tempo.
    """
    if not SKLEARN_OK:
        raise RuntimeError("sklearn/KMeans indisponível no ambiente.")
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

    # ===== Mapping estável pelos CENTROIDES (valor_m2 do centro) =====
    # Inverte escala dos centroides para a escala real
    centers_real = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers_real, columns=feats)

    # Ordena clusters pelo valor_m2 do centróide
    order = centers_df.sort_values("valor_m2").index.tolist()

    labels = ["01 - Original", "02 - Semi", "03 - Reformado"]
    mapping = {cid: labels[min(i // 3, 2)] for i, cid in enumerate(order)}
    return scaler, km, mapping

def aplicar_cluster_fixo(
    dados: pd.DataFrame,
    scaler: StandardScaler,
    km: KMeans,
    mapping: Dict[int, str]
) -> pd.DataFrame:
    """
    Aplica scaler.transform + kmeans.predict e mapeia para cluster_nome fixo.
    """
    feats = ["valor_m2", "area_util"]
    dfc = dados.dropna(subset=feats).copy()
    if dfc.empty:
        return dfc

    X = scaler.transform(dfc[feats])
    dfc["cluster_id"] = km.predict(X)
    dfc["cluster_nome"] = dfc["cluster_id"].map(mapping)
    return dfc

def plot_clusters_scatter(dfc: pd.DataFrame, out_prefix: str) -> None:
    if dfc.empty:
        return

    plt.figure(figsize=(12, 7))
    for nome, seg in dfc.groupby("cluster_nome"):
        plt.scatter(seg["area_util"], seg["valor_m2"], alpha=0.55, s=30, label=nome)
    plt.title(f"Clusters (FIXOS) - {CONFIG['BAIRRO_ALVO']} | {out_prefix}")
    plt.xlabel("Área Útil (m²)")
    plt.ylabel("Valor por m² (R$)")
    plt.legend(title="Cluster (fixo)")
    plt.tight_layout()
    fname = f"{out_prefix}_clusters_scatter.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico salvo: {fname}")

def plot_clusters_por_mes(dfc: pd.DataFrame, out_prefix: str) -> None:
    if dfc.empty or "mes" not in dfc.columns or dfc["mes"].isna().all():
        return

    tab = (
        dfc.groupby(["mes", "cluster_nome"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    bottom = None
    for col in tab.columns:
        if bottom is None:
            plt.bar(tab.index.astype(str), tab[col].values, label=col)
            bottom = tab[col].values.copy()
        else:
            plt.bar(tab.index.astype(str), tab[col].values, bottom=bottom, label=col)
            bottom += tab[col].values

    plt.xticks(rotation=45, ha="right")
    plt.title(f"Clusters por mês (FIXOS) - {CONFIG['BAIRRO_ALVO']} | {out_prefix}")
    plt.xlabel("Mês (YYYY-MM)")
    plt.ylabel("Quantidade")
    plt.legend(title="Cluster (fixo)")
    plt.tight_layout()
    fname = f"{out_prefix}_clusters_por_mes.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico salvo: {fname}")

def resumo_clusters(dfc: pd.DataFrame) -> pd.DataFrame:
    if dfc.empty:
        return pd.DataFrame()

    out = (
        dfc.groupby("cluster_nome")
        .agg(
            amostra=("cluster_nome", "size"),
            m2_medio=("valor_m2", "mean"),
            m2_mediana=("valor_m2", "median"),
            area_media=("area_util", "mean"),
        )
        .reset_index()
    )
    out["preco_medio_proxy"] = out["m2_mediana"] * out["area_media"]
    return out.sort_values("m2_medio")

# ============================================================
# Análise completa por janela
# ============================================================

def analisar_janela(
    dados_janela: pd.DataFrame,
    mes_alvo: str,
    inicio: date,
    fim: date,
    scaler: Optional[StandardScaler],
    km: Optional[KMeans],
    mapping: Optional[Dict[int, str]],
) -> None:
    out_prefix = f"janela_{mes_alvo}_({inicio.isoformat()}_a_{fim.isoformat()})".replace("-", "")
    out_prefix = out_prefix.replace("(", "").replace(")", "").replace("__", "_").replace(" ", "")

    print("\n" + "#" * 90)
    print(f"ANÁLISE (JANELA 3 MESES) | MÊS-ALVO={mes_alvo} | {inicio} -> {fim}")
    print("#" * 90)

    geral = [resumo_estatistico(dados_janela, f"{CONFIG['BAIRRO_ALVO']} - GERAL")]
    imprimir_tabela("RESUMO GERAL (janela 3 meses, após limpeza)", geral)

    if CONFIG["ANALISAR_POR_QUARTOS"] and "quartos" in dados_janela.columns:
        rows = []
        for q in sorted(dados_janela["quartos"].unique()):
            if q <= 0:
                continue
            seg = dados_janela[dados_janela["quartos"] == q]
            rows.append(resumo_estatistico(seg, f"{q} quarto(s)"))
        imprimir_tabela("RESUMO POR QUARTOS (janela 3 meses)", rows)

    if CONFIG["ANALISAR_POR_VAGA"] and "vagas" in dados_janela.columns:
        rows = [
            resumo_estatistico(dados_janela[dados_janela["vagas"] <= 0], "Sem vaga"),
            resumo_estatistico(dados_janela[dados_janela["vagas"] > 0], "Com vaga"),
        ]
        imprimir_tabela("RESUMO POR VAGA (janela 3 meses)", rows)

    agg_mes = plot_valores_por_mes(dados_janela, out_prefix)

    # ===== Cluster fixo: aplica (não treina aqui) =====
    dfc = pd.DataFrame()
    cl_resumo = pd.DataFrame()
    if CONFIG["CLUSTERS_ATIVOS"] and scaler is not None and km is not None and mapping is not None:
        dfc = aplicar_cluster_fixo(dados_janela, scaler, km, mapping)

        plot_clusters_scatter(dfc, out_prefix)
        plot_clusters_por_mes(dfc, out_prefix)

        cl_resumo = resumo_clusters(dfc)

        if not cl_resumo.empty:
            out = cl_resumo.copy()
            for c in ["m2_medio", "m2_mediana", "preco_medio_proxy"]:
                out[c] = out[c].map(lambda x: f"{x:,.0f}".replace(",", "."))
            out["area_media"] = out["area_media"].map(lambda x: f"{x:,.1f}".replace(",", "."))

            print("\n" + "=" * 80)
            print("CLUSTERS (FIXOS — treinados uma vez no período global)")
            print("=" * 80)
            print(out.to_string(index=False))

    # Exports (CSV)
    if not agg_mes.empty:
        agg_mes.to_csv(f"{out_prefix}_resumo_mensal.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] CSV salvo: {out_prefix}_resumo_mensal.csv")

    if not cl_resumo.empty:
        cl_resumo.to_csv(f"{out_prefix}_resumo_clusters.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] CSV salvo: {out_prefix}_resumo_clusters.csv")

    print("\n" + "=" * 80)
    print("CHECKLIST (janela 3 meses)")
    print("=" * 80)
    print(f"Amostra final: {len(dados_janela)}")
    if len(dados_janela) > 0:
        print(f"m² mediana: {dados_janela['valor_m2'].median():,.0f}".replace(",", "."))
        print(f"m² média:   {dados_janela['valor_m2'].mean():,.0f}".replace(",", "."))
        print(f"Preço mediana: {dados_janela['preco'].median():,.0f}".replace(",", "."))
        print(f"Área mediana:  {dados_janela['area_util'].median():,.1f}".replace(",", "."))

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Análise por janela (3 meses) via Postgres com CLUSTERS FIXOS (treino global + aplicação nas janelas)."
    )
    parser.add_argument(
        "--meses",
        nargs="*",
        default=CONFIG["MESES_ALVO"],
        help="Meses-alvo YYYY-MM. Ex: --meses 2025-11 2025-12 2026-01",
    )
    args = parser.parse_args()
    meses_alvo = args.meses

    # calcula janelas
    janelas = []
    for ym in meses_alvo:
        ini, fim = janela_3_meses_para_mes_alvo(ym)
        janelas.append((ym, ini, fim))

    inicio_global = min(x[1] for x in janelas)
    fim_global = max(x[2] for x in janelas)

    # ===== intervalo de treino pode ser expandido =====
    treino_ini = add_months(inicio_global, -int(CONFIG["CLUSTER_TREINO_EXTRA_MESES_ANTES"]))
    treino_fim = add_months(fim_global, int(CONFIG["CLUSTER_TREINO_EXTRA_MESES_DEPOIS"]))

    print(f"[INFO] Buscando no banco (GLOBAL): {treino_ini} -> {treino_fim}")
    df = carregar_do_banco(treino_ini, treino_fim)

    if df.empty:
        print("Nenhum dado retornou do banco nesse intervalo.")
        return

    df_limpo = limpar_dados_db(df)
    if df_limpo.empty:
        print("Nenhum dado válido após a limpeza (verifique bairro/tipo/oferta/faixas).")
        return

    # ===== Treina modelo global UMA vez =====
    scaler = None
    km = None
    mapping = None

    if CONFIG["CLUSTERS_ATIVOS"]:
        try:
            scaler, km, mapping = treinar_modelo_cluster_global(df_limpo)
            print("[INFO] Modelo de cluster GLOBAL treinado (fixo no tempo).")
        except Exception as e:
            print(f"[WARN] Falha ao treinar cluster global: {e}")
            scaler = km = mapping = None

    # ===== roda análises das janelas =====
    for (ym, ini, fim) in janelas:
        dados_janela = df_limpo[(df_limpo["data_coleta"] >= ini) & (df_limpo["data_coleta"] <= fim)].copy()
        if dados_janela.empty:
            print(f"[WARN] Janela vazia para {ym} ({ini} -> {fim}).")
            continue

        analisar_janela(dados_janela, ym, ini, fim, scaler, km, mapping)

if __name__ == "__main__":
    main()