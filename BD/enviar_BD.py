# -*- coding: utf-8 -*-
import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


def _env(*nomes: str, obrigatorio: bool = True, padrao: str = "") -> str:
    """Primeira variavel de ambiente definida entre `nomes`.

    Aceita DB_* e PG* porque os scripts do repositorio usaram os dois nomes.
    Sem fallback com credencial: senha em codigo fica no historico do git para
    sempre, mesmo depois de removida do arquivo.
    """
    for nome in nomes:
        valor = os.getenv(nome)
        if valor:
            return valor
    if obrigatorio:
        raise RuntimeError(
            "Variavel de ambiente ausente: " + " ou ".join(nomes) +
            ". Defina no .env (nao versionado) antes de rodar."
        )
    return padrao


def db_config() -> dict:
    """Lido na hora da conexao, nao no import: modulo importavel sem .env."""
    return {
        "host": _env("DB_HOST", "PGHOST"),
        "port": int(_env("DB_PORT", "PGPORT", obrigatorio=False, padrao="5432")),
        "dbname": _env("DB_NAME", "PGDATABASE"),
        "user": _env("DB_USER", "PGUSER"),
        "password": _env("DB_PASSWORD", "PGPASSWORD"),
    }

TABELA_DESTINO = "imoveis"
PORTAL = "df"

CARGAS = [
    {"arquivo": r"./2026-04-05 (1).csv", "data_coleta": "2026-04-05"},
    {"arquivo": r"./2026-04-12.csv", "data_coleta":"2026-04-12"},
    {"arquivo": r"./2026-04-19.csv", "data_coleta":"2026-04-19"},
    {"arquivo": r"./2026-04-26.csv", "data_coleta":"2026-04-26"},
]


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return df


def tratar_valores(df: pd.DataFrame, data_coleta: str, portal: str) -> pd.DataFrame:
    df = df.copy()
    df = normalizar_colunas(df)

    # O CSV novo possui id e horario.
    # id será ignorado.
    # horario será usado apenas para gerar data_coleta.
    # link pode não existir no CSV e será criado como NULL.
    colunas_esperadas_csv = [
        "codigo",
        "creci",
        "anunciante",
        "oferta",
        "tipo",
        "area_util",
        "bairro",
        "cidade",
        "preco",
        "valor_m2",
        "quartos",
        "vagas",
        "latitude",
        "longitude",
        "quadra",
        "horario",
        "link",
    ]

    for col in colunas_esperadas_csv:
        if col not in df.columns:
            df[col] = None

    colunas_numericas = [
        "area_util",
        "preco",
        "valor_m2",
        "quartos",
        "vagas",
        "latitude",
        "longitude",
    ]

    for col in colunas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    colunas_texto = [
        "cidade",
        "codigo",
        "link",
        "creci",
        "anunciante",
        "oferta",
        "tipo",
        "quadra",
        "bairro",
    ]

    for col in colunas_texto:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

    # Pega somente a data da coluna horario.
    # Exemplo: "2026-04-05 13:42:10" vira "2026-04-05".
    df["data_coleta"] = pd.to_datetime(
        df["horario"],
        errors="coerce"
    ).dt.date

    # Se alguma linha vier sem horario válido,
    # usa a data informada manualmente em CARGAS como fallback.
    df["data_coleta"] = df["data_coleta"].fillna(
        pd.to_datetime(data_coleta).date()
    )

    df["portal"] = portal

    df_final = df[
        [
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
            "portal",
        ]
    ].copy()

    # Converte NaN, NaT e <NA> para None, que vira NULL no PostgreSQL.
    df_final = df_final.astype(object)
    df_final = df_final.where(pd.notnull(df_final), None)

    for col in colunas_texto:
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(
                lambda x: None
                if x is None or str(x).strip() == "" or str(x) == "<NA>"
                else x
            )

    return df_final


def inserir_dataframe(conn, df: pd.DataFrame):
    if df.empty:
        print("DataFrame vazio. Nada para inserir.")
        return

    colunas_insert = [
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
        "portal",
    ]

    df = df[colunas_insert].copy()

    registros = [
        tuple(None if pd.isna(valor) else valor for valor in row)
        for row in df.itertuples(index=False, name=None)
    ]

    query = sql.SQL("""
        INSERT INTO {tabela} ({colunas})
        VALUES %s
    """).format(
        tabela=sql.Identifier(TABELA_DESTINO),
        colunas=sql.SQL(", ").join(map(sql.Identifier, colunas_insert)),
    )

    with conn.cursor() as cur:
        execute_values(
            cur,
            query.as_string(conn),
            registros,
            page_size=1000,
        )

    conn.commit()
    print(f"{len(df)} linhas inseridas com sucesso.")


def processar_cargas():
    cfg = db_config()
    if not cfg["password"]:
        raise ValueError(
            "A variável de ambiente DB_PASSWORD não foi definida. "
            "Defina a senha antes de rodar o script."
        )

    conn = psycopg2.connect(**cfg)

    try:
        for carga in CARGAS:
            arquivo = carga["arquivo"]
            data_coleta = carga["data_coleta"]

            print("\n" + "=" * 60)
            print(f"Lendo arquivo: {arquivo}")
            print("Data da coleta será extraída da coluna 'horario'.")
            print(f"Data fallback: {data_coleta}")

            if not os.path.exists(arquivo):
                print(f"Arquivo não encontrado: {arquivo}")
                continue

            df = pd.read_csv(
                arquivo,
                encoding="utf-8-sig",
                low_memory=False,
            )

            print(f"Colunas encontradas no CSV: {list(df.columns)}")
            print(f"Total de linhas no CSV: {len(df)}")

            df_tratado = tratar_valores(
                df=df,
                data_coleta=data_coleta,
                portal=PORTAL,
            )

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