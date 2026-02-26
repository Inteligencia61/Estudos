# =========================
# Apartamento_Asa_Sul.py  (EXECUTA VIA IMPORTLIB)
# - Mantém padrão do seu Sheet: NomeMedia | Vaga | Valor | AmostraAnalisada | Data
# - Adiciona também médias por QUADRA (igual metragem/quartos/bairro)
# =========================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from google.oauth2 import service_account
from googleapiclient.discovery import build


# =========================
# Utils
# =========================
def remover_outliers_iqr(df, coluna):
    s = df[coluna].dropna()
    if s.empty:
        return df
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0 or pd.isna(IQR):
        return df
    filtro = (df[coluna] >= (Q1 - 1.5 * IQR)) & (df[coluna] <= (Q3 + 1.5 * IQR))
    return df[filtro]


def selecionar_clusters(cluster_means, ordered_clusters):
    semi_reformado_idx = len(ordered_clusters) // 2
    semi_reformado_cluster = ordered_clusters[semi_reformado_idx]
    semi_reformado_value = cluster_means[semi_reformado_cluster]

    original_cluster = None
    reformado_cluster = None

    for cluster in ordered_clusters:
        if cluster_means[cluster] <= semi_reformado_value * 0.90:
            original_cluster = cluster
        if cluster_means[cluster] >= semi_reformado_value * 1.10 and reformado_cluster is None:
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


def grupos_metragem_quartos(df, tipo_imovel):
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

    df["grupo_metragem"] = pd.cut(df["area_util"], bins=metragem_bins, labels=metragem_labels)
    df["quartos_group"] = pd.cut(df["quartos"], bins=quartos_bins, labels=quartos_labels)


def escolher_valor_coluna(df_filtrado, tipo_imovel):
    if tipo_imovel == "Apartamento":
        return "valor_m2" if "valor_m2" in df_filtrado.columns else "preco"
    return "preco"


# =========================
# Core
# =========================
def analisar_imovel_detalhado(df, tipo_imovel=None, bairro=None, cidade=None, quadra=None):
    resultados_finais = []

    # padroniza texto
    if "bairro" in df.columns:
        df["bairro"] = df["bairro"].astype(str).str.strip().str.upper()
    if "cidade" in df.columns:
        df["cidade"] = df["cidade"].astype(str).str.strip().str.upper()
    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip()
    if "quadra" in df.columns:
        df["quadra"] = df["quadra"].astype(str).str.strip().str.upper()

    # numéricos
    for col in ["preco", "valor_m2", "area_util", "quartos", "vagas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # calcula valor_m2 se vier 0
    if all(c in df.columns for c in ["valor_m2", "preco", "area_util"]):
        mask = (df["valor_m2"] <= 0) & (df["preco"] > 0) & (df["area_util"] > 0)
        df.loc[mask, "valor_m2"] = df.loc[mask, "preco"] / df.loc[mask, "area_util"]

    if "vagas" not in df.columns:
        df["vagas"] = 0.0

    # agrupa por vaga
    for vaga_status, df_vaga in df.groupby(df["vagas"] > 0):
        vaga_status_str = "Com Vaga" if vaga_status else "Sem Vaga"

        filtro = pd.Series(True, index=df_vaga.index)

        if tipo_imovel and "tipo" in df_vaga.columns:
            filtro &= (df_vaga["tipo"] == tipo_imovel)
        if bairro and "bairro" in df_vaga.columns:
            filtro &= (df_vaga["bairro"] == bairro)
        if cidade and "cidade" in df_vaga.columns:
            filtro &= (df_vaga["cidade"] == cidade)
        if quadra and "quadra" in df_vaga.columns:
            filtro &= (df_vaga["quadra"] == quadra)

        df_filtrado = df_vaga[filtro].copy()
        if df_filtrado.empty:
            continue

        valor_coluna = escolher_valor_coluna(df_filtrado, tipo_imovel)

        # remove outliers
        if valor_coluna in df_filtrado.columns:
            df_filtrado = remover_outliers_iqr(df_filtrado, valor_coluna)

        # KMeans
        if (not df_filtrado.empty) and (valor_coluna in df_filtrado.columns) and (len(df_filtrado) >= 9):
            X = df_filtrado[[valor_coluna]].values
            kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
            df_filtrado["cluster"] = kmeans.fit_predict(X)
            df_filtrado["cluster"] = df_filtrado["cluster"].astype("category")

            cluster_means = (
                df_filtrado.groupby("cluster", observed=False)[valor_coluna]
                .mean()
                .sort_values()
            )
            ordered_clusters = cluster_means.index.tolist()

            cluster_labels = selecionar_clusters(cluster_means, ordered_clusters)
            df_filtrado["cluster_nomeado"] = df_filtrado["cluster"].map(cluster_labels)

            df_cluster_ok = df_filtrado.dropna(subset=["cluster_nomeado"])
            media_clusters = df_cluster_ok.groupby("cluster_nomeado").agg(
                Valor=(valor_coluna, "mean"),
                AmostraAnalisada=("cluster_nomeado", "size"),
            ).reset_index().assign(NomeMedia=lambda x: x["cluster_nomeado"])
        else:
            media_clusters = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])

        # grupos metragem/quartos
        if "area_util" not in df_filtrado.columns:
            df_filtrado["area_util"] = 0.0
        if "quartos" not in df_filtrado.columns:
            df_filtrado["quartos"] = 0.0

        grupos_metragem_quartos(df_filtrado, tipo_imovel)

        media_metragem = df_filtrado.groupby("grupo_metragem", dropna=False, observed=False).agg(
            Valor=(valor_coluna, "mean"),
            AmostraAnalisada=("grupo_metragem", "size"),
        ).reset_index().assign(
            NomeMedia=lambda x: x["grupo_metragem"].apply(lambda y: f"Metragem {y}")
        )

        media_quartos = df_filtrado.groupby("quartos_group", dropna=False, observed=False).agg(
            Valor=(valor_coluna, "mean"),
            AmostraAnalisada=("quartos_group", "size"),
        ).reset_index().assign(
            NomeMedia=lambda x: x["quartos_group"].apply(lambda y: f"Quartos {y}")
        )

        # bairro
        media_bairros = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])
        if "bairro" in df_filtrado.columns:
            media_bairros = df_filtrado.groupby("bairro").agg(
                Valor=(valor_coluna, "mean"),
                AmostraAnalisada=("bairro", "size"),
            ).reset_index().assign(NomeMedia=lambda x: x["bairro"].apply(lambda y: f"Bairro {y}"))

        # ✅ QUADRA (NOVO) - todas as quadras encontradas
        media_quadras = pd.DataFrame(columns=["NomeMedia", "Valor", "AmostraAnalisada"])
        if "quadra" in df_filtrado.columns:
            # remove quadras vazias tipo "" / "NAN"
            df_q = df_filtrado.copy()
            df_q["quadra"] = df_q["quadra"].astype(str).str.strip().str.upper()
            df_q.loc[df_q["quadra"].isin(["", "NAN", "NONE"]), "quadra"] = np.nan

            media_quadras = df_q.dropna(subset=["quadra"]).groupby("quadra").agg(
                Valor=(valor_coluna, "mean"),
                AmostraAnalisada=("quadra", "size"),
            ).reset_index().assign(NomeMedia=lambda x: x["quadra"].apply(lambda y: f"Quadra {y}"))

        resultados = pd.concat(
            [media_clusters, media_metragem, media_quartos, media_bairros, media_quadras],
            ignore_index=True,
        )
        resultados["Vaga"] = vaga_status_str
        resultados_finais.append(resultados)

    if not resultados_finais:
        return pd.DataFrame(columns=["NomeMedia", "Vaga", "Valor", "AmostraAnalisada"])

    resultados_final = pd.concat(resultados_finais, ignore_index=True)
    return resultados_final[["NomeMedia", "Vaga", "Valor", "AmostraAnalisada"]]


def executar_analise_com_data(input_file, data_ref, tipo_imovel="Apartamento", bairro=None, cidade=None, quadra=None):
    df = pd.read_csv(input_file)

    resultados = analisar_imovel_detalhado(
        df=df,
        tipo_imovel=tipo_imovel,
        bairro=bairro,
        cidade=cidade,
        quadra=quadra,
    )

    # ✅ padrão do seu sheet: "Data" no final
    resultados["Data"] = pd.to_datetime(data_ref).date().isoformat()

    # ✅ padrão de colunas
    return resultados[["NomeMedia", "Vaga", "Valor", "AmostraAnalisada", "Data"]]


def salvar_no_google_sheets(df, spreadsheet_id, range_name, credentials_json_path, incluir_cabecalho=False):
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    creds = service_account.Credentials.from_service_account_file(
        credentials_json_path,
        scopes=SCOPES,
    )

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

# parâmetros do script
RANGE_NAME = "Apartamento_Asa_Sul!A1"
TIPO_IMOVEL = "Apartamento"
BAIRRO = "ASA SUL"

print(f"[INFO] Rodando {BAIRRO} | Data={data_ref} | arquivo={input_file}")

resultados_finais = executar_analise_com_data(
    input_file=input_file,
    data_ref=data_ref,
    tipo_imovel=TIPO_IMOVEL,
    bairro=BAIRRO,
    cidade=None,
    quadra=None,
)

# ✅ não manda cabeçalho (para não bagunçar a planilha)
salvar_no_google_sheets(
    df=resultados_finais,
    spreadsheet_id=spreadsheet_id,
    range_name=RANGE_NAME,
    credentials_json_path=credentials_json,
    incluir_cabecalho=False,
)
