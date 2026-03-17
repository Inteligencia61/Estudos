# -*- coding: utf-8 -*-
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DB_CONFIG = {
    "host": "db-restore.ctug6oqcsj14.us-east-2.rds.amazonaws.com",
    "port": 5432,
    "dbname": "coleta_imobiliaria",
    "user": "inteligencia",
    "password": "61imoveis"
}

TABELA_DESTINO = "imoveis"

CARGAS = [
    {"arquivo": r"./2026-02-01 (1).csv", "data_coleta": "2026-02-01"},
    {"arquivo": r"./2026-02-08.csv", "data_coleta": "2026-02-08"},
    {"arquivo": r"./2026-02-15.csv", "data_coleta": "2026-02-15"},
    {"arquivo": r"./2026-02-23.csv", "data_coleta": "2026-02-23"},
    {"arquivo": r"./2026-03-02.csv", "data_coleta": "2026-03-02"},
]

def tratar_valores(df: pd.DataFrame, data_coleta: str) -> pd.DataFrame:
    df = df.copy()

    colunas_esperadas_csv = [
        "codigo", "link", "creci", "anunciante", "oferta", "tipo",
        "area_util", "bairro", "cidade", "preco", "valor_m2",
        "quartos", "vagas", "latitude", "longitude", "quadra"
    ]

    for col in colunas_esperadas_csv:
        if col not in df.columns:
            df[col] = None

    colunas_numericas = [
        "area_util", "preco", "valor_m2", "quartos",
        "vagas", "latitude", "longitude"
    ]

    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    colunas_texto = [
        "cidade", "codigo", "link", "creci", "anunciante",
        "oferta", "tipo", "quadra", "bairro"
    ]

    for col in colunas_texto:
        df[col] = df[col].astype(object)

    df["data_coleta"] = pd.to_datetime(data_coleta).date()

    df_final = df[[
        "area_util",
        "preco",
        "valor_m2",
        "quartos",
        "vagas",
        "latitude",
        "longitude",
        "data_coleta",
        "cidade",
        "codigo",
        "link",
        "creci",
        "anunciante",
        "oferta",
        "tipo",
        "quadra",
        "bairro",
    ]].copy()

    df_final = df_final[df_final["codigo"].notna()].copy()
    df_final = df_final.astype(object)
    df_final = df_final.where(pd.notnull(df_final), None)

    antes = len(df_final)
    df_final = df_final.drop_duplicates(subset=["codigo", "data_coleta"], keep="first").copy()
    removidos = antes - len(df_final)

    if removidos > 0:
        print(f"Duplicados removidos dentro do CSV: {removidos}")

    return df_final

def buscar_chaves_existentes(conn, data_coleta):
    sql = f"""
        SELECT codigo, data_coleta
        FROM {TABELA_DESTINO}
        WHERE data_coleta = %s
    """
    df_existentes = pd.read_sql(sql, conn, params=[data_coleta])

    if df_existentes.empty:
        return set()

    return set(
        (str(row["codigo"]) if row["codigo"] is not None else None, row["data_coleta"])
        for _, row in df_existentes.iterrows()
    )

def filtrar_ja_existentes(conn, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    data_ref = df["data_coleta"].iloc[0]
    existentes = buscar_chaves_existentes(conn, data_ref)

    if not existentes:
        return df

    mask = df.apply(
        lambda row: (str(row["codigo"]) if row["codigo"] is not None else None, row["data_coleta"]) not in existentes,
        axis=1
    )

    removidos = len(df) - mask.sum()
    if removidos > 0:
        print(f"Linhas ignoradas por já existirem no banco: {removidos}")

    return df[mask].copy()

def inserir_dataframe(conn, df: pd.DataFrame):
    if df.empty:
        print("DataFrame vazio. Nada para inserir.")
        return

    registros = [
        tuple(None if pd.isna(valor) else valor for valor in row)
        for row in df.itertuples(index=False, name=None)
    ]

    sql = f"""
        INSERT INTO {TABELA_DESTINO} (
            area_util,
            preco,
            valor_m2,
            quartos,
            vagas,
            latitude,
            longitude,
            data_coleta,
            cidade,
            codigo,
            link,
            creci,
            anunciante,
            oferta,
            tipo,
            quadra,
            bairro
        )
        VALUES %s
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, registros, page_size=1000)

    conn.commit()

def processar_cargas():
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        for carga in CARGAS:
            arquivo = carga["arquivo"]
            data_coleta = carga["data_coleta"]

            print(f"\nLendo arquivo: {arquivo}")
            print(f"Data da coleta: {data_coleta}")

            df = pd.read_csv(arquivo, low_memory=False)
            df_tratado = tratar_valores(df, data_coleta)
            df_tratado = filtrar_ja_existentes(conn, df_tratado)

            print(f"Total de linhas para inserir: {len(df_tratado)}")
            inserir_dataframe(conn, df_tratado)

            print("Carga concluída com sucesso.")

    except Exception as e:
        conn.rollback()
        print(f"Erro durante a carga: {e}")
        raise

    finally:
        conn.close()

if __name__ == "__main__":
    processar_cargas()