# -*- coding: utf-8 -*-

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


def normalizar_username(nome):
    if pd.isna(nome):
        return ""

    nome = str(nome).strip()
    nome = re.sub(r"\s+", " ", nome)
    return nome.replace(" ", "_")


def encontrar_coluna(df, candidatos):
    colunas_mapa = {c.lower().strip(): c for c in df.columns}

    for candidato in candidatos:
        chave = candidato.lower().strip()
        if chave in colunas_mapa:
            return colunas_mapa[chave]

    raise ValueError(
        "Não encontrei nenhuma das colunas esperadas: {}\nColunas disponíveis: {}".format(
            candidatos, list(df.columns)
        )
    )


def montar_dataframe_usuarios(df):
    col_nome = encontrar_coluna(df, ["Nome", "nome"])
    col_id_corretor = encontrar_coluna(df, ["IdCorretor", "idCorretor", "id_corretor"])
    col_id_gerente = encontrar_coluna(df, ["IdGerente", "idGerente", "id_gerente"])

    base = df[[col_nome, col_id_corretor, col_id_gerente]].copy()
    base.columns = ["nome", "id_usuarios", "team"]

    base = base.dropna(subset=["nome", "id_usuarios", "team"]).copy()

    base["nome"] = base["nome"].astype(str).str.strip()
    base["id_usuarios"] = base["id_usuarios"].astype(str).str.strip()
    base["team"] = base["team"].astype(str).str.strip()

    base["username"] = base["nome"].apply(normalizar_username)

    duplicados = base["username"].duplicated(keep=False)
    base.loc[duplicados, "username"] = (
        base.loc[duplicados, "username"] + "_" + base.loc[duplicados, "id_usuarios"]
    )

    base["password"] = "12345678"
    base["permissao"] = "user"
    base["email"] = ""
    base["telefone"] = ""
    base["instagram"] = ""
    base["descricao"] = ""

    base = base[
        [
            "username",
            "password",
            "team",
            "permissao",
            "id_usuarios",
            "nome",
            "email",
            "telefone",
            "instagram",
            "descricao",
        ]
    ].copy()

    return base


def salvar_csv(df, caminho_saida):
    df.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    print("CSV gerado com sucesso em: {}".format(caminho_saida))


def enviar_para_api(df, base_url, timeout=120, pausa=0.2):
    url = base_url.rstrip("/") + "/auth/cadastro"

    sucesso = 0
    falha = 0

    session = requests.Session()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        payload = {
            "username": row["username"],
            "password": row["password"],
            "team": row["team"],
            "permissao": row["permissao"],
            "id_usuarios": row["id_usuarios"],
            "nome": row["nome"],
            "email": row["email"],
            "telefone": row["telefone"],
            "instagram": row["instagram"],
            "descricao": row["descricao"],
        }

        print("[{}/{}] Enviando {}".format(i, len(df), payload["username"]))

        try:
            response = session.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=timeout,
            )

            try:
                resposta_json = response.json()
            except Exception:
                resposta_json = response.text

            if response.ok:
                sucesso += 1
                print("[OK] {} cadastrado com sucesso.".format(payload["username"]))
            else:
                falha += 1
                print(
                    "[ERRO] {} | status={} | resposta={}".format(
                        payload["username"], response.status_code, resposta_json
                    )
                )

        except requests.exceptions.ReadTimeout:
            falha += 1
            print("[TIMEOUT] {} demorou mais que {}s".format(payload["username"], timeout))
        except Exception as e:
            falha += 1
            print("[ERRO] {} | exceção: {}".format(payload["username"], e))

        time.sleep(pausa)

    print("\nResumo do envio:")
    print("Sucessos: {}".format(sucesso))
    print("Falhas: {}".format(falha))


def main():
    parser = argparse.ArgumentParser(
        description="Gera usuários a partir da base de corretores e opcionalmente cadastra na API."
    )
    parser.add_argument("--input", required=True, help="Caminho do CSV de entrada")
    parser.add_argument(
        "--output",
        default="usuarios_corretores_para_cadastro.csv",
        help="Caminho do CSV de saída",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:5000",
        help="URL base da API",
    )
    parser.add_argument(
        "--somente-gerar-csv",
        action="store_true",
        help="Se informado, apenas gera o CSV",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout de cada requisição em segundos",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=0.2,
        help="Pausa entre requisições em segundos",
    )

    args = parser.parse_args()

    caminho_entrada = Path(args.input)
    if not caminho_entrada.exists():
        raise FileNotFoundError("Arquivo não encontrado: {}".format(caminho_entrada))

    df = pd.read_csv(str(caminho_entrada))
    usuarios_df = montar_dataframe_usuarios(df)

    salvar_csv(usuarios_df, args.output)

    if not args.somente_gerar_csv:
        enviar_para_api(
            usuarios_df,
            args.base_url,
            timeout=args.timeout,
            pausa=args.pausa,
        )


if __name__ == "__main__":
    main()