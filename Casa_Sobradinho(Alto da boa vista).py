# =========================
# Casa_Sobradinho(Alto da boa vista).py
# (EXECUTA VIA IMPORTLIB)
# - Envia para o Sheets no padrão EXATO:
#   NomeMedia | Vaga | Valor | AmostraAnalisada | Data
# =========================

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from google.oauth2 import service_account
from googleapiclient.discovery import build


# =========================================================
# Utilidades
# =========================================================
def remover_outliers_iqr(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    if coluna not in df.columns or df.empty:
        return df
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    if pd.isna(IQR) or IQR == 0:
        return df
    filtro = (df[coluna] >= (Q1 - 1.5 * IQR)) & (df[coluna] <= (Q3 + 1.5 * IQR))
    return df.loc[filtro].copy()


def selecionar_clusters(cluster_means: pd.Series, ordered_clusters: list) -> dict:
    semi_reformado_idx = len(ordered_clusters) // 2
    semi_reformado_cluster = ordered_clusters[semi_reformado_idx]
    semi_reformado_value = float(cluster_means[semi_reformado_cluster])

    original_cluster = None
    reformado_cluster = None

    for cluster in ordered_clusters:
        if float(cluster_means[cluster]) <= semi_reformado_value * 0.90:
            original_cluster = cluster
        if float(cluster_means[cluster]) >= semi_reformado_value * 1.10 and reformado_cluster is None:
            reformado_cluster = cluster

    if original_cluster is None:
        original_cluster = ordered_clusters[0]
    if reformado_cluster is None:
        reformado_cluster = ordered_clusters[-1]

    return {
        original_cluster: "01 - Original",
        semi_reformado_cluster: "02 - Semi-Reformado",
        reformado_cluster: "03 - Reformado",
    }


def grupos_metragem_quartos(df: pd.DataFrame, tipo_imovel: str) -> None:
    if "area_util" not in df.columns:
        df["area_util"] = 0.0
    if "quartos" not in df.columns:
        df["quartos"] = 0.0

    if tipo_imovel == "Apartamento":
        metragem_bins = [0, 50, 75, 90, 130, 160, 200, np.inf]
        metragem_labels = ["<50", "50-75", "75-90", "90-130", "130-160", "160-200", ">200"]
        quartos_bins = [0, 1, 2, 3, 4, np.inf]
        quartos_labels = ["1", "2", "3", "4", "+5"]
    else:
        metragem_bins = [0, 400, 600, 800, 1000, np.inf]
        metragem_labels = ["<400", "400-600", "600-800", "800-1000", ">1000"]
        quartos_bins = [0, 4, np.inf]
        quartos_labels = ["Até 4", "5 ou mais"]

    df["grupo_metragem"] = pd.cut(df["area_util"], bins=metragem_bins, labels=metragem_labels, include_lowest=True)
    df["quartos_group"] = pd.cut(df["quartos"], bins=quartos_bins, labels=quartos_labels, include_lowest=True)


def _padrao_por_valor(df: pd.DataFrame, valor_coluna: str, valor_limite: float | None) -> pd.Series:
    if valor_limite is None:
        return pd.Series(["N/A"] * len(df), index=df.index)
    return np.where(df[valor_coluna] >= valor_limite, "Alto Padrão", "Padrão")


def analisar_imovel_detalhado(
    df: pd.DataFrame,
    oferta: str,
    tipo_imovel: str,
    bairro: str | None = None,
    cidade: str | None = None,
    cep: str | None = None,
    valor_limite: float | None = None,
) -> pd.DataFrame:
    resultados_finais = []

    if "vagas" not in df.columns:
        df["vagas"] = 0.0
    df["vagas"] = pd.to_numeric(df["vagas"], errors="coerce").fillna(0.0)

    # Separa com/sem vaga
    for vaga_status, df_vaga in df.groupby(df["vagas"] > 0):
        vaga_status_str = "Com Vaga" if bool(vaga_status) else "Sem Vaga"

        filtro = (df_vaga.get("oferta") == oferta)
        if tipo_imovel:
            filtro &= (df_vaga.get("tipo") == tipo_imovel)
        if bairro:
            filtro &= (df_vaga.get("bairro") == bairro)
        if cidade:
            filtro &= (df_vaga.get("cidade") == cidade)
        if cep:
            filtro &= (df_vaga.get("cep") == cep)

        df_filtrado = df_vaga.loc[filtro].copy()

        valor_coluna = "valor_m2" if tipo_imovel == "Apartamento" else "preco"
        if valor_coluna not in df_filtrado.columns:
            # sem coluna principal: retorna vazio padronizado
            return pd.DataFrame(columns=["NomeMedia", "Vaga", "Padrão", "Valor", "AmostraAnalisada"])

        df_filtrado = remover_outliers_iqr(df_filtrado, valor_coluna)

        # Padrão (opcional)
        df_filtrado["Padrão"] = _padrao_por_valor(df_filtrado, valor_coluna, valor_limite)

        # ---- Clusters (se amostra suficiente)
        if (not df_filtrado.empty) and (len(df_filtrado) >= 9):
            X = df_filtrado[[valor_coluna]].values
            kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
            df_filtrado["cluster"] = kmeans.fit_predict(X).astype("int")
            df_filtrado["cluster"] = df_filtrado["cluster"].astype("category")

            cluster_means = (
                df_filtrado.groupby("cluster", observed=False)[valor_coluna]
                .mean()
                .sort_values()
            )
            ordered_clusters = cluster_means.index.tolist()

            cluster_labels = selecionar_clusters(cluster_means, ordered_clusters)
            df_filtrado["cluster_nomeado"] = df_filtrado["cluster"].map(cluster_labels)

            media_clusters = (
                df_filtrado.dropna(subset=["cluster_nomeado"])
                .groupby("cluster_nomeado", observed=False)
                .agg(
                    Valor=(valor_coluna, "mean"),
                    AmostraAnalisada=("cluster_nomeado", "size"),
                )
                .reset_index()
                .rename(columns={"cluster_nomeado": "NomeMedia"})
            )
        else:
            media_clusters = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        # ---- Metragem e Quartos
        if "area_util" not in df_filtrado.columns:
            df_filtrado["area_util"] = 0.0
        if "quartos" not in df_filtrado.columns:
            df_filtrado["quartos"] = 0.0

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

        # ---- Bairro/Cidade
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

        if "cidade" in df_filtrado.columns:
            media_cidades = (
                df_filtrado.groupby("cidade", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("cidade", "size"))
                .reset_index()
            )
            media_cidades["NomeMedia"] = media_cidades["cidade"].apply(
                lambda y: f"Cidade {y}" if pd.notna(y) else "Cidade Sem Cidade"
            )
            media_cidades = media_cidades[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_cidades = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        # ---- Quadras
        if "quadra" in df_filtrado.columns:
            quadra_norm = df_filtrado["quadra"].astype(str).str.strip()
            quadra_norm = quadra_norm.replace({"nan": "", "None": ""})
            df_filtrado["_quadra_norm"] = np.where(quadra_norm.eq(""), "Sem Quadra", quadra_norm)

            media_quadras = (
                df_filtrado.groupby("_quadra_norm", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("_quadra_norm", "size"))
                .reset_index()
                .rename(columns={"_quadra_norm": "quadra"})
            )
            media_quadras["NomeMedia"] = media_quadras["quadra"].apply(lambda y: f"Quadra {y}")
            media_quadras = media_quadras[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_quadras = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        # ---- CEP
        if "cep" in df_filtrado.columns:
            media_ceps = (
                df_filtrado.groupby("cep", dropna=False, observed=False)
                .agg(Valor=(valor_coluna, "mean"), AmostraAnalisada=("cep", "size"))
                .reset_index()
            )
            media_ceps["NomeMedia"] = media_ceps["cep"].apply(
                lambda y: f"CEP {y}" if pd.notna(y) else "CEP Sem CEP"
            )
            media_ceps = media_ceps[["NomeMedia", "Valor", "AmostraAnalisada"]]
        else:
            media_ceps = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        resultados = pd.concat(
            [media_clusters, media_metragem, media_quartos, media_bairros, media_cidades, media_quadras, media_ceps],
            ignore_index=True,
        )
        resultados["Vaga"] = vaga_status_str

        # Padrão como coluna do resultado (se houver mistura, mantemos "Misto")
        if "Padrão" in df_filtrado.columns and not df_filtrado.empty:
            padrao_unicos = df_filtrado["Padrão"].dropna().unique().tolist()
            resultados["Padrão"] = padrao_unicos[0] if len(padrao_unicos) == 1 else "Misto"
        else:
            resultados["Padrão"] = "N/A"

        resultados_finais.append(resultados)

    if not resultados_finais:
        return pd.DataFrame(columns=["NomeMedia", "Vaga", "Padrão", "Valor", "AmostraAnalisada"])

    resultados_final = pd.concat(resultados_finais, ignore_index=True)
    return resultados_final[["NomeMedia", "Vaga", "Padrão", "Valor", "AmostraAnalisada"]]


def salvar_no_google_sheets(
    resultados: pd.DataFrame,
    spreadsheet_id: str,
    range_name: str,
    credentials_json_path: str,
    incluir_cabecalho: bool = False,
    sheet_columns: list[str] | None = None,
) -> None:
    """
    Envia para o Google Sheets garantindo:
    - ordem e quantidade de colunas EXATAS do Sheet
    - evita deslocamento de colunas
    """
    if sheet_columns is None:
        sheet_columns = ["NomeMedia", "Vaga", "Valor", "AmostraAnalisada", "Data"]

    resultados = resultados.copy()

    # garante colunas do sheet
    for col in sheet_columns:
        if col not in resultados.columns:
            resultados[col] = ""

    # mantém SOMENTE as colunas do sheet, na ordem certa
    resultados = resultados[sheet_columns]

    # NaN: textos -> "", números -> 0
    numeric_cols = set(["Valor", "AmostraAnalisada"])
    for col in resultados.columns:
        if col in numeric_cols:
            resultados[col] = pd.to_numeric(resultados[col], errors="coerce").fillna(0)
        else:
            resultados[col] = resultados[col].astype(str).replace({"nan": "", "None": ""}).fillna("")

    valores = resultados.values.tolist()
    if incluir_cabecalho:
        valores = [resultados.columns.tolist()] + valores

    with open(credentials_json_path, "r", encoding="utf-8") as f:
        credentials_info = json.load(f)

    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    service = build("sheets", "v4", credentials=credentials)
    sheet = service.spreadsheets()

    corpo_requisicao = {"values": valores}

    request = sheet.values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=corpo_requisicao,
    )
    response = request.execute()
    print(f"{response.get('updates', {}).get('updatedCells', 0)} células atualizadas.")


def _read_input(input_file: str) -> pd.DataFrame:
    ext = os.path.splitext(input_file)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(input_file)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(input_file)
    raise ValueError(f"Formato não suportado: {ext}")


# =========================================================
# Config do script (padrão do seu Sheet = 5 colunas)
# =========================================================
# Recomendo travar no tamanho real do padrão:
range_name = "Casa_Sobradinho(Alto da boa vista)!A:E"

# Estas variáveis vêm do seu runner via importlib:
# - input_file
# - credentials_json
# - spreadsheet_id
# - data_ref
try:
    input_file  # noqa: F401
    credentials_json  # noqa: F401
    spreadsheet_id  # noqa: F401
except NameError as e:
    raise RuntimeError(
        "Este script deve ser executado pelo runner que injeta input_file, credentials_json e spreadsheet_id."
    ) from e

data_ref_value = globals().get("data_ref", None)  # string "YYYY-MM-DD" (recomendado)

df = _read_input(input_file)

# Se a base tiver 'data', filtramos por data_ref; se não tiver, roda com a base toda
if "data" in df.columns:
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    if data_ref_value is None:
        raise RuntimeError("data_ref não foi informado no runner (ex: data_ref='2026-02-01').")
    data_ref_dt = pd.to_datetime(data_ref_value, errors="coerce")
    if pd.isna(data_ref_dt):
        raise RuntimeError(f"data_ref inválido: {data_ref_value}")
    df = df.loc[df["data"] == data_ref_dt].copy()

# Numéricos essenciais
for col in ["preco", "valor_m2", "area_util", "quartos", "vagas"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df[["preco", "area_util", "quartos", "vagas"]] = (
    df.reindex(columns=["preco", "area_util", "quartos", "vagas"]).fillna(0.0)
)

# Rodar análise
resultados = analisar_imovel_detalhado(
    df=df,
    oferta="Publicado",
    tipo_imovel="Casa",
    bairro="ALTO DA BOA VISTA",
    cidade=None,
    cep=None,
    valor_limite=None,
)

# Data vai na última coluna do padrão do Sheet
resultados["Data"] = str(data_ref_value) if data_ref_value is not None else ""

# Enviar ao Sheets no padrão EXATO:
# NomeMedia | Vaga | Valor | AmostraAnalisada | Data
salvar_no_google_sheets(
    resultados=resultados,
    spreadsheet_id=spreadsheet_id,
    range_name=range_name,
    credentials_json_path=credentials_json,
    incluir_cabecalho=False,
    sheet_columns=["NomeMedia", "Vaga", "Valor", "AmostraAnalisada", "Data"],
)
