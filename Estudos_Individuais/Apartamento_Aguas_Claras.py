# =========================
# Apartamento_Aguas_Claras.py  (EXECUTA VIA IMPORTLIB)
# - Padrão único do Sheet: NomeMedia | Vaga | Valor | AmostraAnalisada | Data
# - Inclui: clusters (Original/Semi/Reformado), Metragem, Quartos, Bairro, Quadra e (se existir) CEP
# - Compatível com CSV sem coluna "data": a data vem do runner (data_ref)
# =========================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from google.oauth2 import service_account
from googleapiclient.discovery import build


# =========================
# Leitura / Normalização
# =========================
def ler_csv_flex(input_file: str) -> pd.DataFrame:
    """Lê CSV com separador ',' ou ';' (e UTF-8-SIG)."""
    for sep in (",", ";"):
        try:
            df = pd.read_csv(input_file, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(input_file, encoding="utf-8-sig")


def normalizar_texto_series(s: pd.Series) -> pd.Series:
    """Padroniza texto: strip, upper, e converte vazios para NA."""
    s = s.astype("string").str.strip().str.upper()
    s = s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    return s


# =========================
# Regras
# =========================
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
    """Escolhe 3 clusters: Original / Semi / Reformado com base na mediana e limites +-10%."""
    semi_idx = len(ordered_clusters) // 2
    semi_cluster = ordered_clusters[semi_idx]
    semi_val = cluster_means[semi_cluster]

    original_cluster = None
    reformado_cluster = None

    for c in ordered_clusters:
        if cluster_means[c] <= semi_val * 0.90:
            original_cluster = c
        if cluster_means[c] >= semi_val * 1.10 and reformado_cluster is None:
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
    """Cria colunas grupo_metragem e quartos_group."""
    if tipo_imovel == "APARTAMENTO":
        metragem_bins = [0, 50, 75, 90, 130, 160, 200, np.inf]
        metragem_labels = ["<50", "50-75", "75-90", "90-130", "130-160", "160-200", ">200"]
        quartos_bins = [0, 1, 2, 3, 4, np.inf]
        quartos_labels = ["1", "2", "3", "4", "+5"]
    else:
        metragem_bins = [0, 400, 600, 800, 1000, np.inf]
        metragem_labels = ["<400", "400-600", "600-800", "800-1000", ">1000"]
        quartos_bins = [0, 4, np.inf]
        quartos_labels = ["ATÉ 4", "5 OU MAIS"]

    if "area_util" not in df.columns:
        df["area_util"] = 0.0
    if "quartos" not in df.columns:
        df["quartos"] = 0.0

    df["grupo_metragem"] = pd.cut(df["area_util"], bins=metragem_bins, labels=metragem_labels, include_lowest=True)
    df["quartos_group"] = pd.cut(df["quartos"], bins=quartos_bins, labels=quartos_labels, include_lowest=True)


def escolher_valor_coluna(df: pd.DataFrame, tipo_imovel: str) -> str:
    """Para apartamento: prioriza valor_m2, senão usa preco; para casa: preco."""
    if tipo_imovel == "APARTAMENTO":
        return "valor_m2" if "valor_m2" in df.columns else "preco"
    return "preco"


# =========================
# Core
# =========================
def analisar_imovel_detalhado(
    df: pd.DataFrame,
    oferta: str = "VENDA",
    tipo_imovel: str | None = None,
    bairro: str | None = None,
    cidade: str | None = None,
    quadra: str | None = None,
) -> pd.DataFrame:
    """
    Retorna: NomeMedia | Vaga | Valor | AmostraAnalisada
    """
    df = df.copy()

    # Normalização texto (só se existir)
    for c in ["oferta", "tipo", "bairro", "cidade", "cep", "quadra"]:
        if c in df.columns:
            df[c] = normalizar_texto_series(df[c])

    # Numéricos
    for c in ["preco", "valor_m2", "area_util", "quartos", "vagas"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["preco", "valor_m2", "area_util", "quartos", "vagas"]:
        if c not in df.columns:
            df[c] = 0.0
    df[["preco", "valor_m2", "area_util", "quartos", "vagas"]] = df[
        ["preco", "valor_m2", "area_util", "quartos", "vagas"]
    ].fillna(0.0)

    # calcula valor_m2 quando possível
    if all(c in df.columns for c in ["valor_m2", "preco", "area_util"]):
        mask = (df["valor_m2"] <= 0) & (df["preco"] > 0) & (df["area_util"] > 0)
        df.loc[mask, "valor_m2"] = df.loc[mask, "preco"] / df.loc[mask, "area_util"]

    # Filtro base
    filtro = pd.Series(True, index=df.index)

    # oferta: aceita VENDA / PUBLICADO (se CSV vier assim)
    if "oferta" in df.columns:
        oferta_up = (oferta or "").upper()
        filtro &= df["oferta"].isin([oferta_up, "PUBLICADO", "VENDA"])

    if tipo_imovel and "tipo" in df.columns:
        filtro &= (df["tipo"] == tipo_imovel.upper())

    if bairro and "bairro" in df.columns:
        filtro &= (df["bairro"] == bairro.upper())

    if cidade and "cidade" in df.columns:
        filtro &= (df["cidade"] == cidade.upper())

    if quadra and "quadra" in df.columns:
        filtro &= (df["quadra"] == quadra.upper())

    df = df.loc[filtro].copy()
    if df.empty:
        return pd.DataFrame(columns=["NomeMedia", "Vaga", "Valor", "AmostraAnalisada"])

    resultados_finais: list[pd.DataFrame] = []

    # Agrupa por Vaga (0 => Sem Vaga)
    for vaga_status, df_vaga in df.groupby(df["vagas"] > 0, observed=False):
        vaga_status_str = "Com Vaga" if vaga_status else "Sem Vaga"

        df_filtrado = df_vaga.copy()
        if df_filtrado.empty:
            continue

        valor_coluna = escolher_valor_coluna(df_filtrado, (tipo_imovel or "").upper() if tipo_imovel else "APARTAMENTO")

        # Outliers
        if valor_coluna in df_filtrado.columns:
            df_filtrado = remover_outliers_iqr(df_filtrado, valor_coluna)

        # =========================
        # Clusters (3 grupos)
        # =========================
        if (not df_filtrado.empty) and (valor_coluna in df_filtrado.columns) and (len(df_filtrado) >= 9):
            X = df_filtrado[[valor_coluna]].values
            kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
            df_filtrado["cluster"] = kmeans.fit_predict(X)
            df_filtrado["cluster"] = df_filtrado["cluster"].astype("category")

            cluster_means = df_filtrado.groupby("cluster", observed=False)[valor_coluna].mean().sort_values()
            ordered = cluster_means.index.tolist()
            labels = selecionar_clusters(cluster_means, ordered)

            df_filtrado["cluster_nomeado"] = df_filtrado["cluster"].map(labels)
            df_cluster_ok = df_filtrado.dropna(subset=["cluster_nomeado"]).copy()

            media_clusters = (
                df_cluster_ok.groupby("cluster_nomeado", observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("cluster_nomeado", "size"))
                .reset_index()
                .rename(columns={"cluster_nomeado": "NomeMedia"})
            )
        else:
            media_clusters = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        # =========================
        # Metragem / Quartos
        # =========================
        grupos_metragem_quartos(df_filtrado, (tipo_imovel or "APARTAMENTO").upper())

        media_metragem = (
            df_filtrado.groupby("grupo_metragem", dropna=False, observed=False)
            .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("grupo_metragem", "size"))
            .reset_index()
        )
        media_metragem["NomeMedia"] = media_metragem["grupo_metragem"].apply(lambda y: f"Metragem {y}")
        media_metragem = media_metragem[["NomeMedia", "Valor", "AmostraAnalisada"]]

        media_quartos = (
            df_filtrado.groupby("quartos_group", dropna=False, observed=False)
            .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("quartos_group", "size"))
            .reset_index()
        )
        media_quartos["NomeMedia"] = media_quartos["quartos_group"].apply(lambda y: f"Quartos {y}")
        media_quartos = media_quartos[["NomeMedia", "Valor", "AmostraAnalisada"]]

        # =========================
        # Bairro (se existir)
        # =========================
        if "bairro" in df_filtrado.columns:
            media_bairros = (
                df_filtrado.groupby("bairro", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("bairro", "size"))
                .reset_index()
            )
            media_bairros["NomeMedia"] = media_bairros["bairro"].apply(lambda y: f"Bairro {y}")
            media_bairros = media_bairros[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_bairros = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        # =========================
        # Quadra (todas)
        # =========================
        if "quadra" in df_filtrado.columns:
            df_q = df_filtrado.copy()
            df_q["quadra"] = normalizar_texto_series(df_q["quadra"])
            media_quadras = (
                df_q.groupby("quadra", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("quadra", "size"))
                .reset_index()
            )
            media_quadras["NomeMedia"] = media_quadras["quadra"].apply(lambda y: f"Quadra {y}")
            media_quadras = media_quadras[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_quadras = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        # =========================
        # CEP (se existir no CSV)
        # =========================
        if "cep" in df_filtrado.columns:
            df_c = df_filtrado.copy()
            df_c["cep"] = df_c["cep"].astype("string").str.strip()
            df_c.loc[df_c["cep"].isin(["", "nan", "NONE", "NULL", "NaN", "NAN"]), "cep"] = pd.NA
            media_ceps = (
                df_c.groupby("cep", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("cep", "size"))
                .reset_index()
            )
            media_ceps["NomeMedia"] = media_ceps["cep"].apply(lambda y: f"CEP {y}")
            media_ceps = media_ceps[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_ceps = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        resultados = pd.concat(
            [media_clusters, media_metragem, media_quartos, media_bairros, media_quadras, media_ceps],
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
    tipo_imovel: str,
    oferta: str,
    bairro: str | None,
    cidade: str | None,
    quadra: str | None,
) -> pd.DataFrame:
    df = ler_csv_flex(input_file)

    resultados = analisar_imovel_detalhado(
        df=df,
        oferta=oferta,
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
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(credentials_json_path, scopes=SCOPES)
    service = build("sheets", "v4", credentials=creds)

    df = df.copy().fillna("")
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

# =========================
# CONFIG DESTE SCRIPT
# =========================
RANGE_NAME = "Apartamento_Aguas_Claras!A1"
TIPO_IMOVEL = "Apartamento"
OFERTA = "Venda"
BAIRRO = None
CIDADE = "ÁGUAS CLARAS"
QUADRA = None

print(f"[INFO] Rodando Apartamento_Aguas_Claras.py | Data={data_ref} | arquivo={input_file}")

resultados_finais = executar_analise_com_data(
    input_file=input_file,
    data_ref=data_ref,
    tipo_imovel=TIPO_IMOVEL,
    oferta=OFERTA,
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
