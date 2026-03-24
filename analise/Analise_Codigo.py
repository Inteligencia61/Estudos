
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Opcional: clusters
try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "BAIRRO_ALVO": "LAGO SUL",
    "TIPO_ALVO": "CASA",
    "OFERTA_ALVO": "Publicado",

    "PRECO_MIN": 1_000_000,
    "PRECO_MAX": 30_000_000,
    "AREA_MIN": 40,
    "AREA_MAX": 500_000,

    "VLM2_MIN": 6_000,
    "VLM2_MAX": 900_000,

    "APLICAR_IQR": True,

    "ANALISAR_POR_QUARTOS": True,
    "ANALISAR_POR_VAGA": True,

    "CLUSTERS_ATIVOS": True,
    "MIN_AMOSTRA_PARA_CLUSTER": 20,

    # Nova segmentação
    "MIN_AMOSTRA_PARA_CLUSTER_LUXO": 12,
}


# ============================================================
# Leitura e normalização
# ============================================================

def ler_csv_flex(path: str) -> pd.DataFrame:
    """Tenta ler CSV com ',' ou ';' e encoding utf-8-sig."""
    for sep in (",", ";"):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8-sig")


def normalizar_texto(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    return s


def to_number(s: pd.Series) -> pd.Series:
    """
    Converte string para número:
    - remove pontos de milhar
    - troca ',' por '.'
    """
    s = s.astype("string")
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def extrair_data_do_nome_arquivo(path: str) -> Optional[str]:
    """Extrai YYYY-MM-DD do nome do arquivo. Ex: 2026-01-18.csv"""
    base = os.path.basename(path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", base)
    return m.group(1) if m else None


def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    col_map = {c: re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    aliases = {
        "codigo": ["codigo", "código", "cod", "id_anuncio", "idanuncio", "id"],
        "bairro": ["bairro", "setor", "regiao", "região"],
        "cidade": ["cidade", "municipio", "município"],
        "tipo": ["tipo", "tipo_imovel", "tipologia"],
        "oferta": ["oferta", "finalidade", "negocio", "negócio"],
        "preco": ["preco", "preço", "valor", "valor_total", "valorvenda", "preco_venda"],
        "area_util": ["area_util", "area", "área", "area_privativa", "areautil", "m2"],
        "quartos": ["quartos", "dormitorios", "dormitórios", "dorms"],
        "vagas": ["vagas", "garagens", "vaga", "qtde_vagas"],
        "valor_m2": ["valor_m2", "valor_por_m2", "vlm2", "preco_m2", "preco_por_m2"],
        "quadra": ["quadra", "quad", "q"],
        "cep": ["cep"],
        "data": ["data", "data_coleta", "dt", "date"],
    }

    rename_to_std = {}
    cols = set(df.columns)
    for std, al in aliases.items():
        for a in al:
            if a in cols:
                rename_to_std[a] = std
                break

    df.rename(columns=rename_to_std, inplace=True)
    return df


# ============================================================
# Limpeza
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


def limpar_dados(df: pd.DataFrame, data_ref: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df = padronizar_colunas(df)

    for c in ["bairro", "cidade", "tipo", "oferta", "quadra", "cep"]:
        if c in df.columns:
            df[c] = normalizar_texto(df[c])

    for c in ["preco", "area_util", "quartos", "vagas", "valor_m2"]:
        if c not in df.columns:
            df[c] = np.nan

    df["preco"] = to_number(df["preco"])
    df["area_util"] = to_number(df["area_util"])
    df["quartos"] = to_number(df["quartos"])
    df["vagas"] = to_number(df["vagas"])
    df["valor_m2"] = to_number(df["valor_m2"])

    # data: coluna > nome do arquivo
    if "data" in df.columns:
        dt = pd.to_datetime(df["data"], errors="coerce")
        df["data"] = dt.dt.date.astype("string")
    else:
        df["data"] = data_ref if data_ref else pd.NA

    # remove duplicados
    #if "codigo" in df.columns:
    #    df["codigo"] = normalizar_texto(df["codigo"])
    #    df = df.drop_duplicates(subset=["codigo"])
    #else:
    #   df = df.drop_duplicates(subset=["preco", "area_util", "quartos", "bairro", "tipo"])

    # filtros alvo
    if "bairro" in df.columns:
        df = df[df["bairro"] == CONFIG["BAIRRO_ALVO"]]
    if "tipo" in df.columns:
        df = df[df["tipo"] == CONFIG["TIPO_ALVO"]]
    if "oferta" in df.columns:
        df = df[df["oferta"].isin([CONFIG["OFERTA_ALVO"], "PUBLICADO", "VENDA"])]

    # remove inválidos
    df = df.dropna(subset=["preco", "area_util"])
    df = df[(df["preco"] >= CONFIG["PRECO_MIN"]) & (df["preco"] <= CONFIG["PRECO_MAX"])]
    df = df[(df["area_util"] >= CONFIG["AREA_MIN"]) & (df["area_util"] <= CONFIG["AREA_MAX"])]

    # calcula valor_m2 sempre
    df["valor_m2"] = df["preco"] / df["area_util"]

    # faixa de mercado
    df = df[(df["valor_m2"] >= CONFIG["VLM2_MIN"]) & (df["valor_m2"] <= CONFIG["VLM2_MAX"])]

    # IQR
    if CONFIG["APLICAR_IQR"] and len(df) >= 20:
        df = remover_outliers_iqr(df, "valor_m2")

    df["quartos"] = df["quartos"].fillna(0).astype(int)
    df["vagas"] = df["vagas"].fillna(0).astype(int)

    df["data_dt"] = pd.to_datetime(df["data"], errors="coerce")
    df["mes"] = df["data_dt"].dt.to_period("M").astype("string")

    return df


# ============================================================
# Análises (prints)
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
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
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
# Gráficos por mês
# ============================================================

def plot_mensal_base(dados: pd.DataFrame) -> None:
    base = dados.dropna(subset=["data_dt"]).copy()
    if base.empty or base["mes"].isna().all():
        print("[INFO] Sem datas válidas para gráficos mensais.")
        return

    agg_mes = (
        base.groupby("mes", dropna=False)
        .agg(
            amostra=("mes", "size"),
            m2_mediana=("valor_m2", "median"),
            m2_medio=("valor_m2", "mean"),
        )
        .reset_index()
        .sort_values("mes")
    )

    plt.figure(figsize=(12, 6))
    plt.plot(agg_mes["mes"], agg_mes["m2_mediana"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Valor/m² (mediana) por mês - {CONFIG['BAIRRO_ALVO']}")
    plt.xlabel("Mês (YYYY-MM)")
    plt.ylabel("R$/m² (mediana)")
    for i, row in agg_mes.iterrows():
        plt.text(i, row["m2_mediana"], f" n={int(row['amostra'])}", fontsize=9, va="bottom", ha="center")
    plt.tight_layout()
    plt.savefig("trend_mensal_m2.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Gráfico salvo: trend_mensal_m2.png")

    meses_ordenados = agg_mes["mes"].tolist()
    series = [base.loc[base["mes"] == m, "valor_m2"].values for m in meses_ordenados]

    plt.figure(figsize=(12, 6))
    plt.boxplot(series, labels=meses_ordenados, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Distribuição do Valor/m² por mês - {CONFIG['BAIRRO_ALVO']}")
    plt.xlabel("Mês (YYYY-MM)")
    plt.ylabel("R$/m²")
    plt.tight_layout()
    plt.savefig("boxplot_mensal_m2.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Gráfico salvo: boxplot_mensal_m2.png")


def plot_diario_se_existir(dados: pd.DataFrame) -> None:
    base = dados.dropna(subset=["data_dt"]).copy()
    if base.empty:
        return

    agg = (
        base.groupby(base["data_dt"].dt.date)
        .agg(amostra=("valor_m2", "size"), m2_mediana=("valor_m2", "median"))
        .reset_index()
        .sort_values("data_dt")
    )

    if len(agg) < 2:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(agg["data_dt"], agg["m2_mediana"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Valor/m² (mediana) por data de coleta - {CONFIG['BAIRRO_ALVO']}")
    plt.xlabel("Data")
    plt.ylabel("R$/m² (mediana)")
    plt.tight_layout()
    plt.savefig("trend_diaria_m2.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Gráfico salvo: trend_diaria_m2.png")


# ============================================================
# Clusters padrão
# ============================================================

def clusters_3_grupos(dados: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not SKLEARN_OK or not CONFIG.get("CLUSTERS_ATIVOS", False):
        return None
    if dados.empty or len(dados) < CONFIG.get("MIN_AMOSTRA_PARA_CLUSTER", 20):
        print("[INFO] Amostra pequena para cluster (ou clusters desativados).")
        return None

    colunas_cluster = ["valor_m2", "area_util"]
    df_clean = dados.dropna(subset=colunas_cluster).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[colunas_cluster])

    km = KMeans(n_clusters=9, random_state=42, n_init=10)
    df_clean["cluster_id"] = km.fit_predict(X_scaled)

    cluster_order = (
        df_clean.groupby("cluster_id")["valor_m2"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    labels = ["01 - Original", "02 - Semi", "03 - Reformado"]
    mapping = {cluster_id: labels[min(i // 3, 2)] for i, cluster_id in enumerate(cluster_order)}
    df_clean["cluster_nome"] = df_clean["cluster_id"].map(mapping)

    plt.figure(figsize=(12, 7))
    for nome, seg in df_clean.groupby("cluster_nome"):
        plt.scatter(seg["area_util"], seg["valor_m2"], alpha=0.55, s=30, label=nome)
    plt.title(f"Distribuição de Clusters - {CONFIG['BAIRRO_ALVO']} (Área x R$/m²)")
    plt.xlabel("Área Útil (m²)")
    plt.ylabel("Valor por m² (R$)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    nome_plot = f"clusters_{CONFIG['BAIRRO_ALVO'].lower().replace(' ', '_')}.png"
    plt.savefig(nome_plot, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico salvo: {nome_plot}")

    if "mes" in df_clean.columns and df_clean["mes"].notna().any():
        tab = (
            df_clean.groupby(["mes", "cluster_nome"])
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
        plt.title(f"Distribuição de Clusters por mês - {CONFIG['BAIRRO_ALVO']}")
        plt.xlabel("Mês (YYYY-MM)")
        plt.ylabel("Quantidade")
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.savefig("clusters_por_mes.png", bbox_inches="tight")
        plt.close()
        print("[INFO] Gráfico salvo: clusters_por_mes.png")

    out = (
        df_clean.groupby("cluster_nome")
        .agg(
            amostra=("cluster_nome", "size"),
            m2_medio=("valor_m2", "mean"),
            m2_mediana=("valor_m2", "median"),
            area_media=("area_util", "mean"),
        )
        .reset_index()
    )
    out["preco_medio"] = out["m2_mediana"] * out["area_media"]
    return out.sort_values("m2_medio")


# ============================================================
# Nova clusterização: Luxo e Alto Luxo
# ============================================================

def clusterizar_luxo_alto_luxo(dados: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Clusterização específica em 2 grupos:
    - Luxo
    - Alto Luxo

    Usa preço, valor_m2 e área_util para separar os imóveis.
    O cluster com menor preço médio vira Luxo.
    O cluster com maior preço médio vira Alto Luxo.
    """
    if not SKLEARN_OK or not CONFIG.get("CLUSTERS_ATIVOS", False):
        return None

    if dados.empty or len(dados) < CONFIG.get("MIN_AMOSTRA_PARA_CLUSTER_LUXO", 12):
        print("[INFO] Amostra pequena para cluster de Luxo/Alto Luxo.")
        return None

    colunas_cluster = ["preco", "valor_m2", "area_util"]
    df_clean = dados.dropna(subset=colunas_cluster).copy()

    if len(df_clean) < CONFIG.get("MIN_AMOSTRA_PARA_CLUSTER_LUXO", 12):
        print("[INFO] Amostra insuficiente após limpeza para cluster de Luxo/Alto Luxo.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[colunas_cluster])
    n_cluster=100
    km = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    df_clean["cluster_id"] = km.fit_predict(X_scaled)
    metade = n_cluster // 2 

    cluster_order = (
        df_clean.groupby("cluster_id")["preco"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    labels = ["Luxo", "Alto Luxo"]
    mapping = {
        cluster_id: "Luxo" if i < metade else "Alto Luxo"
        for i, cluster_id in enumerate(cluster_order)
    }
    df_clean["cluster_nome"] = df_clean["cluster_id"].map(mapping)

    # Scatter
    plt.figure(figsize=(12, 7))
    for nome, seg in df_clean.groupby("cluster_nome"):
        plt.scatter(seg["area_util"], seg["valor_m2"], alpha=0.60, s=35, label=nome)

    plt.title(f"Clusterização Luxo x Alto Luxo - {CONFIG['BAIRRO_ALVO']}")
    plt.xlabel("Área Útil (m²)")
    plt.ylabel("Valor por m² (R$)")
    plt.legend(title="Segmento")
    plt.tight_layout()
    plt.savefig("clusters_luxo_alto_luxo.png", bbox_inches="tight")
    plt.close()
    print("[INFO] Gráfico salvo: clusters_luxo_alto_luxo.png")

    # Barras por mês
    if "mes" in df_clean.columns and df_clean["mes"].notna().any():
        tab = (
            df_clean.groupby(["mes", "cluster_nome"])
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
        plt.title(f"Luxo x Alto Luxo por mês - {CONFIG['BAIRRO_ALVO']}")
        plt.xlabel("Mês (YYYY-MM)")
        plt.ylabel("Quantidade")
        plt.legend(title="Segmento")
        plt.tight_layout()
        plt.savefig("luxo_alto_luxo_por_mes.png", bbox_inches="tight")
        plt.close()
        print("[INFO] Gráfico salvo: luxo_alto_luxo_por_mes.png")

    out = (
        df_clean.groupby("cluster_nome")
        .agg(
            amostra=("cluster_nome", "size"),
            preco_medio=("preco", "mean"),
            preco_mediana=("preco", "median"),
            m2_medio=("valor_m2", "mean"),
            m2_mediana=("valor_m2", "median"),
            area_media=("area_util", "mean"),
            area_mediana=("area_util", "median"),
        )
        .reset_index()
        .sort_values("preco_medio")
    )

    return out


# ============================================================
# Orquestração
# ============================================================

def carregar_varios_csvs(paths: List[str]) -> pd.DataFrame:
    bases = []
    for p in paths:
        df = ler_csv_flex(p)
        data_ref = extrair_data_do_nome_arquivo(p)
        df_limpo = limpar_dados(df, data_ref=data_ref)
        df_limpo["arquivo_origem"] = os.path.basename(p)
        bases.append(df_limpo)

    if not bases:
        return pd.DataFrame()

    return pd.concat(bases, ignore_index=True)


def analisar_e_printar(dados: pd.DataFrame) -> None:
    print(f"Tamanho df: {len(dados)}")

    geral = [resumo_estatistico(dados, f"{CONFIG['BAIRRO_ALVO']} - GERAL")]
    imprimir_tabela("RESUMO GERAL (após limpeza)", geral)

    # Por quartos
    if CONFIG["ANALISAR_POR_QUARTOS"] and "quartos" in dados.columns:
        rows = []
        for q in sorted(dados["quartos"].dropna().unique()):
            if q <= 0:
                continue
            seg = dados[dados["quartos"] == q]
            rows.append(resumo_estatistico(seg, f"{q} quarto(s)"))
        imprimir_tabela("RESUMO POR QUARTOS", rows)

    # Por vaga
    dados = dados.fillna({"vagas": 0})

    if CONFIG["ANALISAR_POR_VAGA"] and "vagas" in dados.columns:
        rows = []
        seg_sem = dados[dados["vagas"] <= 0]
        seg_com = dados[dados["vagas"] > 0]
        rows.append(resumo_estatistico(seg_sem, "Sem vaga"))
        rows.append(resumo_estatistico(seg_com, "Com vaga"))
        imprimir_tabela("RESUMO POR VAGA", rows)

    # Tendência por data
    if "data_dt" in dados.columns and dados["data_dt"].notna().any():
        trend = (
            dados.dropna(subset=["data_dt"])
            .groupby(dados["data_dt"].dt.date)
            .agg(
                amostra=("valor_m2", "size"),
                m2_mediana=("valor_m2", "median"),
                m2_medio=("valor_m2", "mean"),
            )
            .reset_index()
            .sort_values("data_dt")
        )
        trend["m2_mediana"] = trend["m2_mediana"].map(lambda x: f"{x:,.0f}".replace(",", "."))
        trend["m2_medio"] = trend["m2_medio"].map(lambda x: f"{x:,.0f}".replace(",", "."))

        print("\n" + "=" * 70)
        print("TENDÊNCIA POR DATA (mediana e média do m²)")
        print("=" * 70)
        print(trend.to_string(index=False))

    # Tendência por mês
    if "mes" in dados.columns and dados["mes"].notna().any():
        trend_mes = (
            dados.dropna(subset=["data_dt"])
            .groupby("mes")
            .agg(
                amostra=("mes", "size"),
                m2_mediana=("valor_m2", "median"),
                m2_medio=("valor_m2", "mean"),
            )
            .reset_index()
            .sort_values("mes")
        )
        trend_mes["m2_mediana"] = trend_mes["m2_mediana"].map(lambda x: f"{x:,.0f}".replace(",", "."))
        trend_mes["m2_medio"] = trend_mes["m2_medio"].map(lambda x: f"{x:,.0f}".replace(",", "."))

        print("\n" + "=" * 70)
        print("TENDÊNCIA POR MÊS (mediana e média do m²)")
        print("=" * 70)
        print(trend_mes.to_string(index=False))

    # Gráficos
    plot_diario_se_existir(dados)
    plot_mensal_base(dados)

    # Cluster padrão
    cl = clusters_3_grupos(dados)
    if cl is not None and not cl.empty:
        cl = cl.copy()
        for c in ["m2_medio", "m2_mediana", "preco_medio"]:
            cl[c] = cl[c].map(lambda x: f"{x:,.0f}".replace(",", "."))
        cl["area_media"] = cl["area_media"].map(lambda x: f"{x:,.1f}".replace(",", "."))

        print("\n" + "=" * 70)
        print("CLUSTERS (3 faixas por preço/m²) — proxy de posicionamento")
        print("=" * 70)
        print(cl.to_string(index=False))

    # Novo cluster luxo / alto luxo
    cl_luxo = clusterizar_luxo_alto_luxo(dados)
    if cl_luxo is not None and not cl_luxo.empty:
        cl_luxo = cl_luxo.copy()
        for c in ["preco_medio", "preco_mediana", "m2_medio", "m2_mediana"]:
            cl_luxo[c] = cl_luxo[c].map(lambda x: f"{x:,.0f}".replace(",", "."))
        for c in ["area_media", "area_mediana"]:
            cl_luxo[c] = cl_luxo[c].map(lambda x: f"{x:,.1f}".replace(",", "."))

        print("\n" + "=" * 70)
        print("CLUSTERIZAÇÃO LUXO X ALTO LUXO")
        print("=" * 70)
        print(cl_luxo.to_string(index=False))

    # Checklist
    print("\n" + "=" * 70)
    print("CHECKLIST DE QUALIDADE (sanidade)")
    print("=" * 70)
    print(f"Amostra final: {len(dados)}")
    if len(dados) > 0:
        print(f"m² mediana: {dados['valor_m2'].median():,.0f}".replace(",", "."))
        print(f"m² média:    {dados['valor_m2'].mean():,.0f}".replace(",", "."))
        print(f"Preço médio: {dados['preco'].mean():,.0f}".replace(",", "."))
        print(f"Área média:  {dados['area_util'].mean():,.1f}".replace(",", "."))


def main():
    parser = argparse.ArgumentParser(
        description="Análise limpa e precisa com clusterização Luxo x Alto Luxo."
    )
    parser.add_argument("csvs", nargs="*", help="Lista de arquivos CSV. Ex: 2026-01-04.csv")
    args = parser.parse_args()

    # Se quiser testar direto no código:
    if args.csvs:
        df = carregar_varios_csvs(args.csvs)
    else:
        df = pd.read_csv("./2026-03-08.csv", encoding="utf-8-sig")
        print(len(df))
        df = limpar_dados(df, data_ref="2026-03-08")
        print(len(df))

    if df.empty:
        print("Nenhum dado válido após a limpeza. Verifique bairro/tipo/oferta e faixas.")
        return

    analisar_e_printar(df)


if __name__ == "__main__":
    main()