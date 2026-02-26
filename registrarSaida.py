import re
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build


# =========================
# CONFIG
# =========================
CREDENTIALS_JSON = r"./credentiasl_machome.json"
SPREADSHEET_ID = "1HQDdcbUMj276hnIbPs-WwdWHiUPzMhPRWt4HHRyYGnw"

INPUT_XLSX_PATH = "./saida (2).xlsx"

# IMPORTANTE:
# - Se você deixar None, alguns cenários podem resultar em dict (todas as abas).
# - Para evitar, use 0 (primeira aba) ou o nome exato da aba.
INPUT_SHEET_NAME = 0  # 0 = primeira aba | ou "NomeDaAba" | ou None (vai pegar a primeira automaticamente)

SHEET_SAIDA_NAME = "Fato_Saida"
SHEET_CAPTACAO_NAME = "Fato_Captacao"

DEFAULT_MOTIVO = ""  # vazio = não preencher motivo


# =========================
# Utils
# =========================
def build_sheets_service(credentials_json: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(
        credentials_json, scopes=scopes
    )
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def normalize_codigo(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    # Ex: 8687.0 -> 8687
    if re.fullmatch(r"\d+(\.0+)?", s):
        s = str(int(float(s)))
    return s


def parse_date_any(x) -> str:
    """Retorna data em YYYY-MM-DD. Se vazio -> hoje."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return date.today().strftime("%Y-%m-%d")

    if isinstance(x, (datetime, date)):
        return x.strftime("%Y-%m-%d")

    s = str(x).strip()
    if not s:
        return date.today().strftime("%Y-%m-%d")

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().strftime("%Y-%m-%d")
        except ValueError:
            pass

    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return date.today().strftime("%Y-%m-%d")
        return dt.date().strftime("%Y-%m-%d")
    except Exception:
        return date.today().strftime("%Y-%m-%d")


def sheet_get_values(service, spreadsheet_id: str, range_a1: str) -> List[List[str]]:
    resp = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=range_a1
    ).execute()
    return resp.get("values", [])


def sheet_append_rows(service, spreadsheet_id: str, sheet_name: str, rows: List[List[str]]):
    if not rows:
        return
    body = {"values": rows}
    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()


def read_input_excel(path: str, sheet_name):
    """
    Lê o XLSX e garante DataFrame.
    Se por algum motivo vier dict (todas as abas), pega a primeira aba automaticamente.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    # Mudança principal: se vier dict, pega a primeira aba
    if isinstance(df, dict):
        if not df:
            raise RuntimeError("O Excel não tem abas/dados legíveis.")
        df = list(df.values())[0]

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"Falha ao ler Excel: retorno inesperado ({type(df)}).")

    return df


# =========================
# Main
# =========================
def main():
    service = build_sheets_service(CREDENTIALS_JSON)

    # 1) Ler arquivo de entrada (xlsx com códigos)
    df_in = read_input_excel(INPUT_XLSX_PATH, INPUT_SHEET_NAME)

    # Aceita vários nomes possíveis para a coluna de código
    col_codigo = None
    for c in df_in.columns:
        if str(c).strip().lower() in ("código", "codigo", "code"):
            col_codigo = c
            break
    if col_codigo is None:
        # se não achar, assume a primeira coluna
        col_codigo = df_in.columns[0]

    # DataSaida opcional
    col_data = None
    for c in df_in.columns:
        if str(c).strip().lower() in ("datasaida", "data_saida", "data"):
            col_data = c
            break

    # Motivo opcional
    col_motivo = None
    for c in df_in.columns:
        if str(c).strip().lower() in ("motivo", "motivo_saida", "motivosaida"):
            col_motivo = c
            break

    df_in["__codigo"] = df_in[col_codigo].apply(normalize_codigo)
    df_in = df_in[df_in["__codigo"].notna()].copy()

    if df_in.empty:
        print("Nenhum código válido encontrado no arquivo.")
        return

    # 2) Ler Fato_Captacao do Google Sheets (para resolver Captadores/Gerente)
    cap_values = sheet_get_values(service, SPREADSHEET_ID, f"{SHEET_CAPTACAO_NAME}!A:Z")
    if not cap_values:
        raise RuntimeError(f"Aba {SHEET_CAPTACAO_NAME} está vazia ou não encontrada.")

    cap_header = cap_values[0]
    cap_rows = cap_values[1:]

    def idx(colname: str) -> int:
        try:
            return cap_header.index(colname)
        except ValueError:
            return -1

    i_cod = idx("Código")
    i_c1 = idx("Captador1")
    i_c2 = idx("Captador2")
    i_c3 = idx("Captador3")
    i_g  = idx("Gerente")
    i_dt = idx("DataEntrada")  # usado para pegar o registro mais recente

    if i_cod == -1:
        raise RuntimeError("Não encontrei a coluna 'Código' na aba Fato_Captacao.")

    # mapa: codigo -> (capt1, capt2, capt3, gerente, dataEntrada_raw)
    cap_map: Dict[str, Tuple[str, str, str, str, str]] = {}
    cap_map_dt: Dict[str, datetime] = {}

    for r in cap_rows:
        if i_cod >= len(r):
            continue
        cod = normalize_codigo(r[i_cod])
        if not cod:
            continue

        c1 = r[i_c1] if (i_c1 != -1 and i_c1 < len(r)) else ""
        c2 = r[i_c2] if (i_c2 != -1 and i_c2 < len(r)) else ""
        c3 = r[i_c3] if (i_c3 != -1 and i_c3 < len(r)) else ""
        ge = r[i_g]  if (i_g  != -1 and i_g  < len(r)) else ""

        dt_raw = r[i_dt] if (i_dt != -1 and i_dt < len(r)) else ""
        dt_parsed = None
        if dt_raw:
            # tenta formatos comuns
            try:
                dt_parsed = datetime.strptime(str(dt_raw).strip(), "%d/%m/%Y")
            except Exception:
                try:
                    x = pd.to_datetime(dt_raw, dayfirst=True, errors="coerce")
                    if not pd.isna(x):
                        dt_parsed = x.to_pydatetime()
                except Exception:
                    dt_parsed = None

        dt_key = dt_parsed or datetime.min

        if cod not in cap_map:
            cap_map[cod] = (c1, c2, c3, ge, str(dt_raw))
            cap_map_dt[cod] = dt_key
        else:
            if dt_key >= cap_map_dt[cod]:
                cap_map[cod] = (c1, c2, c3, ge, str(dt_raw))
                cap_map_dt[cod] = dt_key

    # 3) Ler Fato_Saida para evitar duplicar (Código + DataSaida)
    saida_values = sheet_get_values(service, SPREADSHEET_ID, f"{SHEET_SAIDA_NAME}!A:Z")

    if not saida_values:
        existing = set()
    else:
        saida_header = saida_values[0]
        saida_rows = saida_values[1:]

        try:
            s_cod = saida_header.index("Código")
        except ValueError:
            s_cod = 0

        try:
            s_dt = saida_header.index("DataSaida")
        except ValueError:
            s_dt = 6

        existing = set()
        for r in saida_rows:
            cod = normalize_codigo(r[s_cod]) if s_cod < len(r) else None
            dts = r[s_dt] if s_dt < len(r) else ""
            if cod:
                existing.add((cod, str(dts).strip()))

    # 4) Montar linhas a inserir (ordem padrão: Código | Captador1 | Captador2 | Captador3 | Gerente | Motivo | DataSaida)
    rows_to_append: List[List[str]] = []

    for _, row in df_in.iterrows():
        cod = row["__codigo"]

        data_saida = parse_date_any(row[col_data]) if col_data else date.today().strftime("%Y-%m-%d")
        motivo = str(row[col_motivo]).strip() if col_motivo and not pd.isna(row[col_motivo]) else DEFAULT_MOTIVO

        # evita duplicar
        if (cod, data_saida) in existing:
            continue

        c1 = c2 = c3 = ge = ""
        if cod in cap_map:
            c1, c2, c3, ge, _ = cap_map[cod]

        rows_to_append.append([cod, c1, c2, c3, ge, motivo, data_saida])

    if not rows_to_append:
        print("Nada para inserir (talvez já exista tudo na Fato_Saida).")
        return

    # 5) Append
    sheet_append_rows(service, SPREADSHEET_ID, SHEET_SAIDA_NAME, rows_to_append)
    print(f"Inseridos {len(rows_to_append)} registros em {SHEET_SAIDA_NAME}.")


if __name__ == "__main__":
    main()
