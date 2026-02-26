# =========================
# Casa_Jardim_Botanico.py
# (EXECUTA VIA IMPORTLIB)
# - Padrão do Sheet: NomeMedia | Vaga | Valor | AmostraAnalisada | Data
# - Robusto: não quebra se faltarem colunas como cep/quadra/bairro/cidade
# =========================
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from google.oauth2 import service_account
from googleapiclient.discovery import build
# =========================
# Padrão (coluna do Sheet)
# - Se o dataset tiver coluna "padrao"/"padrão"/"PADRAO", usamos ela (moda).
# - Caso contrário, usamos o default abaixo.
# =========================
PADRAO_DEFAULT = "Luxo"

def inferir_padrao(df: pd.DataFrame) -> str:
    for col in ["padrao", "padrão", "PADRAO", "PADRÃO", "Padrao", "Padrão"]:
        if col in df.columns:
            s = df[col].astype("string").str.strip()
            s = s[s.notna() & (s != "")]
            if not s.empty:
                # usa o valor mais frequente
                return s.mode().iloc[0]
    return PADRAO_DEFAULT

# =========================
# Leitura (CSV/Excel)
# =========================
def ler_csv_flex(input_file: str) -> pd.DataFrame:
    for sep in (",", ";"):
        try:
            df = pd.read_csv(input_file, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(input_file, encoding="utf-8-sig")

def read_input_flex(input_file: str) -> pd.DataFrame:
    ext = os.path.splitext(str(input_file))[1].lower()
    if ext in [".csv"]:
        return ler_csv_flex(input_file)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(input_file)
    return ler_csv_flex(input_file)

# =========================
# Utils
# =========================
def normalizar_texto_serie(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "NaN": pd.NA, "NONE": pd.NA})
    return s

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
    mask = (df[coluna] >= (q1 - 1.5 * iqr)) & (df[coluna] <= (q3 + 1.5 * iqr))
    return df.loc[mask].copy()

def selecionar_clusters(cluster_means: pd.Series, ordered_clusters: list) -> dict:
    semi_idx = len(ordered_clusters) // 2
    semi_cluster = ordered_clusters[semi_idx]
    semi_val = cluster_means.loc[semi_cluster]

    original_cluster = None
    reformado_cluster = None

    for c in ordered_clusters:
        if cluster_means.loc[c] <= semi_val * 0.90:
            original_cluster = c
        if (cluster_means.loc[c] >= semi_val * 1.10) and (reformado_cluster is None):
            reformado_cluster = c

    if original_cluster is None:
        original_cluster = ordered_clusters[0]
    if reformado_cluster is None:
        reformado_cluster = ordered_clusters[-1]

    return {
        original_cluster: "01 - Original",
        semi_cluster: "02 - Semi-Reformado",
        reformado_cluster: "03 - Reformado",
    }

def grupos_metragem_quartos(df: pd.DataFrame, tipo_imovel: str) -> None:
    metragem_bins = [0, 400, 600, 800, 1000, np.inf]
    metragem_labels = ["<400", "400-600", "600-800", "800-1000", ">1000"]
    quartos_bins = [0, 4, np.inf]
    quartos_labels = ["Até 4", "5 ou mais"]

    df["grupo_metragem"] = pd.cut(df["area_util"], bins=metragem_bins, labels=metragem_labels, include_lowest=True)
    df["quartos_group"] = pd.cut(df["quartos"], bins=quartos_bins, labels=quartos_labels, include_lowest=True)

def escolher_valor_coluna(df_filtrado: pd.DataFrame) -> str:
    if "preco" in df_filtrado.columns:
        return "preco"
    if "valor_m2" in df_filtrado.columns:
        return "valor_m2"
    return "preco"

# =========================
# Core
# =========================
def analisar_imovel_detalhado(
    df: pd.DataFrame,
    oferta: str = "Venda",
    tipo_imovel: str = "Casa",
    bairro: str | None = None,
    cidade: str | None = None,
    quadra: str | None = None,
) -> pd.DataFrame:
    resultados_finais: list[pd.DataFrame] = []

    for col in ["bairro", "cidade", "tipo", "quadra", "oferta"]:
        if col in df.columns:
            df[col] = normalizar_texto_serie(df[col])


    for col in ["preco", "valor_m2", "area_util", "quartos", "vagas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "vagas" not in df.columns:
        df["vagas"] = 0.0
    if "area_util" not in df.columns:
        df["area_util"] = 0.0
    if "quartos" not in df.columns:
        df["quartos"] = 0.0

    df["vagas"] = pd.to_numeric(df["vagas"], errors="coerce").fillna(0.0)
    df["area_util"] = pd.to_numeric(df["area_util"], errors="coerce").fillna(0.0)
    df["quartos"] = pd.to_numeric(df["quartos"], errors="coerce").fillna(0.0)

    if all(c in df.columns for c in ["valor_m2", "preco", "area_util"]):
        df["valor_m2"] = pd.to_numeric(df["valor_m2"], errors="coerce").fillna(0.0)
        df["preco"] = pd.to_numeric(df["preco"], errors="coerce").fillna(0.0)
        mask = (df["valor_m2"] <= 0) & (df["preco"] > 0) & (df["area_util"] > 0)
        df.loc[mask, "valor_m2"] = df.loc[mask, "preco"] / df.loc[mask, "area_util"]

    for vaga_status, df_vaga in df.groupby(df["vagas"] > 0, observed=False):
        vaga_status_str = "Com Vaga" if bool(vaga_status) else "Sem Vaga"

        filtro = pd.Series(True, index=df_vaga.index)

        if "oferta" in df_vaga.columns and oferta is not None:
            col = df_vaga["oferta"].astype("string").str.strip().str.upper()
            wanted = str(oferta).strip().upper()
            if wanted in set(col.dropna().unique()):
                filtro &= (col == wanted)
        if "tipo" in df_vaga.columns and tipo_imovel:
            filtro &= (df_vaga["tipo"] == str(tipo_imovel).strip().upper())
        if "bairro" in df_vaga.columns and bairro:
            filtro &= (df_vaga["bairro"] == str(bairro).strip().upper())
        if "cidade" in df_vaga.columns and cidade:
            filtro &= (df_vaga["cidade"] == str(cidade).strip().upper())
        if "quadra" in df_vaga.columns and quadra:
            filtro &= (df_vaga["quadra"] == str(quadra).strip().upper())

        df_filtrado = df_vaga.loc[filtro].copy()
        if df_filtrado.empty:
            continue

        valor_coluna = escolher_valor_coluna(df_filtrado)
        if valor_coluna not in df_filtrado.columns:
            continue

        df_filtrado = remover_outliers_iqr(df_filtrado, valor_coluna)

        if (not df_filtrado.empty) and (len(df_filtrado) >= 9):
            X = df_filtrado[[valor_coluna]].values
            kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
            df_filtrado["cluster"] = kmeans.fit_predict(X)
            df_filtrado["cluster"] = df_filtrado["cluster"].astype("category")

            cluster_means = df_filtrado.groupby("cluster", observed=False)[valor_coluna].mean().sort_values()
            ordered = cluster_means.index.tolist()
            labels = selecionar_clusters(cluster_means, ordered)

            df_filtrado["cluster_nomeado"] = df_filtrado["cluster"].map(labels)
            df_cluster_ok = df_filtrado.dropna(subset=["cluster_nomeado"])

            media_clusters = (
                df_cluster_ok.groupby("cluster_nomeado", observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("cluster_nomeado", "size"))
                .reset_index()
                .rename(columns={"cluster_nomeado": "NomeMedia"})
            )
        else:
            media_clusters = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        grupos_metragem_quartos(df_filtrado, tipo_imovel)

        media_metragem = (
            df_filtrado.groupby("grupo_metragem", dropna=False, observed=False)
            .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("grupo_metragem", "size"))
            .reset_index()
        )
        media_metragem["NomeMedia"] = media_metragem["grupo_metragem"].apply(
            lambda y: f"Metragem {y}" if pd.notna(y) else "Metragem Sem Faixa"
        )
        media_metragem = media_metragem[["NomeMedia", "Valor", "AmostraAnalisada"]]

        media_quartos = (
            df_filtrado.groupby("quartos_group", dropna=False, observed=False)
            .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("quartos_group", "size"))
            .reset_index()
        )
        media_quartos["NomeMedia"] = media_quartos["quartos_group"].apply(
            lambda y: f"Quartos {y}" if pd.notna(y) else "Quartos Sem Faixa"
        )
        media_quartos = media_quartos[["NomeMedia", "Valor", "AmostraAnalisada"]]

        if "bairro" in df_filtrado.columns:
            media_bairros = (
                df_filtrado.groupby("bairro", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("bairro", "size"))
                .reset_index()
            )
            media_bairros["NomeMedia"] = media_bairros["bairro"].apply(
                lambda y: f"Bairro {y}" if pd.notna(y) else "Bairro Sem Bairro"
            )
            media_bairros = media_bairros[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_bairros = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        if "quadra" in df_filtrado.columns:
            quadra_norm = df_filtrado["quadra"].astype("string").str.strip().str.upper()
            quadra_norm = quadra_norm.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
            df_filtrado["_quadra_norm"] = quadra_norm

            media_quadras = (
                df_filtrado.groupby("_quadra_norm", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("_quadra_norm", "size"))
                .reset_index()
                .rename(columns={"_quadra_norm": "quadra"})
            )
            media_quadras["NomeMedia"] = media_quadras["quadra"].apply(
                lambda y: f"Quadra {y}" if pd.notna(y) else "Quadra Sem Quadra"
            )
            media_quadras = media_quadras[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_quadras = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        resultados = pd.concat(
            [media_clusters, media_metragem, media_quartos, media_bairros, media_quadras],
            ignore_index=True,
        )
        resultados["Vaga"] = vaga_status_str
        resultados_finais.append(resultados)

    if not resultados_finais:
        return pd.DataFrame(columns=["NomeMedia", "Vaga", "Valor", "AmostraAnalisada"])

    out = pd.concat(resultados_finais, ignore_index=True)
    return out[["NomeMedia", "Vaga", "Valor", "AmostraAnalisada"]]

def executar_analise_com_data(
    input_file: str,
    data_ref: str,
    tipo_imovel: str = "Casa",
    bairro: str | None = None,
    cidade: str | None = None,
    quadra: str | None = None,
) -> pd.DataFrame:
    df = read_input_flex(input_file)

    resultados = analisar_imovel_detalhado(
        df=df,
        oferta="Venda",
        tipo_imovel=tipo_imovel,
        bairro=bairro,
        cidade=cidade,
        quadra=quadra,
    )

    resultados["Data"] = pd.to_datetime(data_ref).date().isoformat()
    return resultados[["NomeMedia", "Vaga", "Valor", "AmostraAnalisada", "Data"]]

def salvar_no_google_sheets(
    df: pd.DataFrame,
    spreadsheet_id: str,
    range_name: str,
    credentials_json_path: str,
    incluir_cabecalho: bool = False,
) -> None:
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(credentials_json_path, scopes=scopes)
    service = build("sheets", "v4", credentials=creds)

    df = df.copy().replace({np.nan: ""})
    if "Data" in df.columns:
        df["Data"] = df["Data"].astype(str)

    values = df.values.tolist()
    if incluir_cabecalho:
        values = [df.columns.tolist()] + values

    resp = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",    
        insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()

    updates = resp.get("updates", {})
    print(
        f"[OK] Sheets append em {range_name} | "
        f"updatedRows={updates.get('updatedRows')} | "
        f"updatedCells={updates.get('updatedCells')}"
    )

# =========================
# EXECUÇÃO AUTOMÁTICA VIA IMPORTLIB
# =========================
try:
    input_file
    credentials_json
    spreadsheet_id
    data_ref
except NameError:
    raise RuntimeError(
        "Este script deve ser executado via importlib com as variáveis globais: "
        "input_file, credentials_json, spreadsheet_id, data_ref"
    )

RANGE_NAME = "Casa_Jardim_Botanico!A1"
TIPO_IMOVEL = "Casa"
BAIRRO = "JARDIM BOTANICO"
CIDADE = None
QUADRA = None

print(f"[INFO] Rodando JARDIM BOTANICO | Data={data_ref} | arquivo={input_file}")

resultados_finais = executar_analise_com_data(
    input_file=input_file,
    data_ref=data_ref,
    tipo_imovel=TIPO_IMOVEL,
    bairro=BAIRRO,
    cidade=CIDADE,
    quadra=QUADRA,
)

salvar_no_google_sheets(
    df=resultados_finais,
    spreadsheet_id=spreadsheet_id,
    range_name=RANGE_NAME,
    credentials_json_path=credentials_json,
    incluir_cabecalho=False,
)