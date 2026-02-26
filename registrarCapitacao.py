# captacao_auto_to_sheets.py
# ============================================================
# Automação 100% em Python -> Google Sheets
#
# O que este script faz:
# 1) Lê uma lista de códigos (XLSX com coluna "Código" OU lista no script)
# 2) Consulta o Imoview (endpoint de VAGOS/DISPONÍVEIS)
#    - Atenção: se o imóvel NÃO estiver vago/disponível, ele pode NÃO retornar,
#      mesmo você passando o código. Por isso criamos um relatório de "faltantes".
# 3) Resolve dimensões no Google Sheets:
#    - Dim_Bairro (cria se não existir)
#    - Dim_Tipo (mapeia por Nome)
#    - Dim_Corretor:
#        * tenta mapear captador por IdImoview -> IdCorretor
#        * se não houver Id no JSON do captador, tenta mapear por Nome (fallback)
#        * se mesmo assim não achar, grava o "id" do Imoview que veio (fallback final)
# 4) Upsert em:
#    - Dim_Imovel: Código | Tipo | Valor | Bairro | Foco PP | Foco AC
#    - Fato_Captacao: Código | Captador1 | Captador2 | Captador3 | Gerente | DataEntrada
# 5) Gera arquivos:
#    - codigos_nao_retornaram.csv   (códigos que estavam no XLSX mas não voltaram do endpoint)
#    - codigos_retornados.csv       (códigos que voltaram)
#
# Requisitos:
#   pip install pandas google-api-python-client google-auth openpyxl requests python-dateutil
# ============================================================

import os
import re
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple

import requests
import pandas as pd
from dateutil.parser import parse as dateparse

from google.oauth2 import service_account
from googleapiclient.discovery import build


# =========================
# CONFIG (EDITE AQUI)
# =========================

# --- Google Sheets ---
CREDENTIALS_JSON = r"./credentiasl_machome.json"  # caminho do JSON do service account
SPREADSHEET_ID = "1HQDdcbUMj276hnIbPs-WwdWHiUPzMhPRWt4HHRyYGnw"

# --- Imoview ---
IMOVIEW_CHAVE = "a4ff7c378eff87533b123d25c9b6f088"  # header: chave

# Endpoint de "imóveis vagos/disponíveis"
IMOVIEW_PATH = "/Imovel/RetornarImoveisDisponiveis"

# --- Finalidade ---
FINALIDADE = 2  # 1=ALUGUEL, 2=VENDA

# --- Como o script vai buscar os imóveis? ---
# Escolha UM modo (deixe os outros vazios)

# MODO 1: Lista de códigos direto no script
CODES_LIST: List[int] = []  # ex: [12345, 67890]

# MODO 2: XLSX com coluna "Código"
CODES_XLSX_PATH = "./entrada.xlsx"  # ex: r"C:\Users\...\entrada_codigos.xlsx"
CODES_XLSX_SHEET = "Página1"

# MODO 3: Janela de data no Imoview (dd/mm/yyyy)
# DATE_FROM = "06/02/2026"
# DATE_TO = "12/02/2026"

# --- Filtros OPCIONAIS da API (deixe None/""/0 para não filtrar) ---
DESTINACAO = 0  # 1=Residencial, 2=Comercial, ... ou 0=Todos
CODIGO_UNIDADE = ""  # ex: "12" ou "12,13" ou "" para todas
SOMENTE_COM_URL_PUBLICA = False  # True/False (opcional)

# --- Paginação ---
NUM_REGISTROS = 20  # máximo 20 conforme doc


# =========================
# CONFIG REGRAS FOCO
# =========================
BAIRROS_PP_RAW = {
    "PLANO PILOTO",
    "ASA SUL",
    "ASA NORTE",
    "NOROESTE",
    "SUDOESTE",
    "JARDIM BOTANICO",
    "LAGO NORTE",
    "LAGO SUL",
}

BAIRROS_AC_RAW = {
    "Águas Claras Norte",
    "Águas Claras Sul",
    "Norte (Águas Claras)",
}

MIN_COMISSAO = 3.5
MIN_VALOR_PP = 1_000_000
MIN_VALOR_AC = 600_000


# =========================
# UTIL
# =========================
def norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = (
        s.replace("Á", "A").replace("À", "A").replace("Ã", "A").replace("Â", "A")
        .replace("É", "E").replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O").replace("Õ", "O").replace("Ô", "O")
        .replace("Ú", "U")
        .replace("Ç", "C")
        .replace("Ä", "A").replace("Ë", "E").replace("Ï", "I").replace("Ö", "O").replace("Ü", "U")
    )
    s = re.sub(r"\s+", " ", s)
    return s


BAIRROS_PP = {norm(x) for x in BAIRROS_PP_RAW}
BAIRROS_AC = {norm(x) for x in BAIRROS_AC_RAW}


def to_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if not s:
        return 0.0

    s = re.sub(r"[^\d,.\-]", "", s)
    if not s or s in {"-", ".", ","}:
        return 0.0

    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return 0.0


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def parse_date_any(x: Any) -> Optional[date]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    s = str(x).strip()
    if not s:
        return None
    try:
        d = dateparse(s, dayfirst=True, fuzzy=True)
        return d.date()
    except Exception:
        return None


def chunk_list(xs: List[int], size: int) -> List[List[int]]:
    out = []
    for i in range(0, len(xs), size):
        out.append(xs[i:i + size])
    return out


# =========================
# IMOVIEW
# =========================
def _get_imoview_chave() -> str:
    chave = os.getenv("IMOVIEW_CHAVE") or IMOVIEW_CHAVE
    if not chave:
        raise RuntimeError("Imoview: defina IMOVIEW_CHAVE (ENV ou seção CONFIG).")
    return chave


def imoview_post(path: str, payload: dict) -> dict:
    chave = _get_imoview_chave()
    url = f"https://api.imoview.com.br{path}"
    headers = {"chave": chave}
    r = requests.post(url, json=payload, headers=headers, timeout=90)
    if not r.ok:
        raise RuntimeError(f"Imoview erro {r.status_code}: {r.text[:1200]}")
    return r.json()


def fetch_imoveis_by_codes(codes: List[int], finalidade: int) -> Dict[int, dict]:
    """
    IMPORTANTE:
    - O endpoint é de VAGOS/DISPONÍVEIS. Se o código não estiver vago/disponível,
      pode NÃO retornar (isso é normal).
    - numeroregistros máximo 20 -> fazemos em lotes.
    """
    codes = sorted(set([int(x) for x in codes if x is not None]))
    if not codes:
        return {}

    out: Dict[int, dict] = {}
    for part in chunk_list(codes, NUM_REGISTROS):
        codes_str = ",".join(str(x) for x in part)
        payload = {
            "finalidade": int(finalidade),
            "numeroPagina": 1,
            "numeroRegistros": int(NUM_REGISTROS),
            "codigosimoveis": codes_str,
            "exibircaptadores": True,
        }

        # filtros opcionais (conforme sua doc)
        if DESTINACAO and int(DESTINACAO) != 0:
            payload["destinacao"] = int(DESTINACAO)
        if CODIGO_UNIDADE:
            payload["codigounidade"] = str(CODIGO_UNIDADE)
        if SOMENTE_COM_URL_PUBLICA:
            payload["somentecomurlpublica"] = True

        resp = imoview_post(IMOVIEW_PATH, payload)
        lista = resp.get("lista") or []
        for item in lista:
            c = to_int(item.get("codigo"))
            if c is not None:
                out[c] = item

    return out


def fetch_imoveis_by_date(date_from_ddmmyyyy: str, date_to_ddmmyyyy: str, finalidade: int) -> Dict[int, dict]:
    page = 1
    out: Dict[int, dict] = {}
    while True:
        payload = {
            "finalidade": int(finalidade),
            "numeroPagina": page,
            "numeroRegistros": int(NUM_REGISTROS),
            # "datacadastroinicio": date_from_ddmmyyyy,
            # "datacadastrofim": date_to_ddmmyyyy,
            "ordenacao": "datainclusaodesc",
            "exibircaptadores": True,
        }

        if DESTINACAO and int(DESTINACAO) != 0:
            payload["destinacao"] = int(DESTINACAO)
        if CODIGO_UNIDADE:
            payload["codigounidade"] = str(CODIGO_UNIDADE)
        if SOMENTE_COM_URL_PUBLICA:
            payload["somentecomurlpublica"] = True

        resp = imoview_post(IMOVIEW_PATH, payload)
        lista = resp.get("lista") or []
        if not lista:
            break

        for item in lista:
            c = to_int(item.get("codigo"))
            if c is not None:
                out[c] = item

        if len(lista) < NUM_REGISTROS:
            break
        page += 1

    return out


def extract_fields(item: dict) -> dict:
    codigo = to_int(item.get("codigo")) or 0

    bairro_nome = (
        item.get("bairro")
        or item.get("nomebairro")
        or item.get("bairronome")
        or (item.get("bairroobj") or {}).get("nome")
        or ""
    )

    valor = (
        item.get("valor")
        or item.get("valorvenda")
        or item.get("preco")
        or item.get("valorimovel")
        or 0
    )

    tipo_nome = (
        item.get("tipo")
        or item.get("nometipo")
        or item.get("tipoimovel")
        or (item.get("tipoobj") or {}).get("nome")
        or ""
    )

    comissao_pct = (
        item.get("taxacomissao")  # aparece no schema
        or item.get("taxaComissao")
        or item.get("comissao")
        or item.get("percentualcomissao")
        or item.get("percentualComissao")
        or item.get("comissaopercentual")
        or 0
    )

    # Pelo schema: datahoracadastro / datahoraultimaalteracao etc.
    # A gente tenta pegar algo que pareça "data de entrada".
    data_entrada = (
        item.get("datahoracadastro")
        or item.get("datacadastro")
        or item.get("dataCadastro")
        or item.get("datainclusao")
        or item.get("dataInclusao")
        or item.get("data")
        or item.get("dataentrada")
        or item.get("dataEntrada")
        or None
    )

    captadores_raw = item.get("captadores") or item.get("listaCaptadores") or item.get("captador") or []

    # Schema oficial não traz id no captador, mas seu retorno anterior trouxe ['46'] etc.
    # Então:
    # - se vier id/codigo -> usamos
    # - se não vier -> usamos nome (para mapear com Dim_Corretor por nome)
    captador_ids: List[str] = []
    captador_nomes: List[str] = []

    if isinstance(captadores_raw, list):
        for c in captadores_raw:
            if isinstance(c, dict):
                cid = (
                    c.get("id")
                    or c.get("idCorretor")
                    or c.get("codigocaptador")
                    or c.get("codigo")
                    or c.get("IdCorretor")
                    or c.get("idimoview")
                    or c.get("IdImoview")
                )
                if cid is not None and str(cid).strip():
                    captador_ids.append(str(cid).strip())

                nome = c.get("nome")
                if nome is not None and str(nome).strip():
                    captador_nomes.append(str(nome).strip())
            else:
                # pode vir direto "46" (string/int)
                captador_ids.append(str(c).strip())

    elif isinstance(captadores_raw, dict):
        cid = (
            captadores_raw.get("id")
            or captadores_raw.get("idCorretor")
            or captadores_raw.get("codigocaptador")
            or captadores_raw.get("codigo")
            or captadores_raw.get("idimoview")
            or captadores_raw.get("IdImoview")
        )
        if cid is not None and str(cid).strip():
            captador_ids.append(str(cid).strip())
        nome = captadores_raw.get("nome")
        if nome is not None and str(nome).strip():
            captador_nomes.append(str(nome).strip())

    # unique preservando ordem
    def unique_keep_order(lst: List[str]) -> List[str]:
        seen = set()
        out2 = []
        for x in lst:
            x = str(x).strip()
            if x and x.lower() != "nan" and x not in seen:
                seen.add(x)
                out2.append(x)
        return out2

    captador_ids = unique_keep_order(captador_ids)[:3]
    captador_nomes = unique_keep_order(captador_nomes)[:3]

    return {
        "codigo": codigo,
        "bairro_nome": str(bairro_nome),
        "valor": to_float(valor),
        "tipo_nome": str(tipo_nome),
        "comissao_pct": to_float(comissao_pct),
        "data_entrada": parse_date_any(data_entrada),
        "captador_ids": captador_ids,
        "captador_nomes": captador_nomes,
    }


# =========================
# REGRAS FOCO
# =========================
def classificar_foco(bairro_nome: str, valor: float, comissao_pct: float, is_residencial: bool) -> Tuple[bool, bool]:
    b = norm(bairro_nome)
    v = float(valor or 0)
    c = float(comissao_pct or 0)
    foco_pp = (b in BAIRROS_PP) and (v >= MIN_VALOR_PP) and (c >= MIN_COMISSAO) and is_residencial
    foco_ac = (b in BAIRROS_AC) and (v >= MIN_VALOR_AC) and (c >= MIN_COMISSAO) and is_residencial
    return foco_pp, foco_ac


# =========================
# GOOGLE SHEETS (API)
# =========================
def sheets_service(credentials_json: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(credentials_json, scopes=scopes)
    return build("sheets", "v4", credentials=creds)


def read_sheet_as_df(svc, spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    rng = f"{sheet_name}!A:Z"
    res = svc.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=rng).execute()
    values = res.get("values", [])
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    fixed = [r + [""] * (len(header) - len(r)) for r in rows]
    return pd.DataFrame(fixed, columns=header)


def write_df_over_sheet(svc, spreadsheet_id: str, sheet_name: str, df: pd.DataFrame):
    values = [list(df.columns)] + df.astype(object).where(pd.notnull(df), "").values.tolist()
    svc.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


# =========================
# MAPAS DAS DIMENSÕES (Sheets)
# =========================
def load_dim_maps(svc, spreadsheet_id: str):
    df_bairro = read_sheet_as_df(svc, spreadsheet_id, "Dim_Bairro")
    df_tipo = read_sheet_as_df(svc, spreadsheet_id, "Dim_Tipo")
    df_corr = read_sheet_as_df(svc, spreadsheet_id, "Dim_Corretor")
    df_imovel = read_sheet_as_df(svc, spreadsheet_id, "Dim_Imovel")
    df_fato = read_sheet_as_df(svc, spreadsheet_id, "Fato_Captacao")

    bairro_name_to_id: Dict[str, str] = {}
    if not df_bairro.empty and "Nome" in df_bairro.columns and "IdBairro" in df_bairro.columns:
        for _, r in df_bairro.iterrows():
            bairro_name_to_id[norm(r.get("Nome", ""))] = str(r.get("IdBairro", "")).strip()

    tipo_name_to_id: Dict[str, str] = {}
    if not df_tipo.empty and "Nome" in df_tipo.columns and "IdTipo" in df_tipo.columns:
        for _, r in df_tipo.iterrows():
            tipo_name_to_id[norm(r.get("Nome", ""))] = str(r.get("IdTipo", "")).strip()

    # Mapas de corretor
    imoview_to_idcorretor: Dict[str, str] = {}     # IdImoview -> IdCorretor (C61xxx)
    nome_to_idcorretor: Dict[str, str] = {}        # Nome -> IdCorretor (fallback se captador não tiver id)
    corretor_to_gerente: Dict[str, str] = {}       # IdCorretor -> IdGerente ou Gerente

    if not df_corr.empty:
        col_idcor = None
        col_idimv = None
        col_nome = None
        col_ger = None

        for c in df_corr.columns:
            nc = norm(c)
            if nc in {"IDCORRETOR", "ID_CORRETOR"}:
                col_idcor = c
            elif nc in {"IDIMOVIEW", "ID_IMOVIEW"}:
                col_idimv = c
            elif nc in {"NOME", "NOMECORRETOR", "CORRETOR"}:
                col_nome = c
            elif nc in {"IDGERENTE", "ID_GERENTE", "GERENTE"}:
                col_ger = c

        # fallbacks comuns
        if col_idcor is None and "IdCorretor" in df_corr.columns:
            col_idcor = "IdCorretor"
        if col_idimv is None and "IdImoview" in df_corr.columns:
            col_idimv = "IdImoview"
        if col_nome is None:
            if "Nome" in df_corr.columns:
                col_nome = "Nome"
            elif "Corretor" in df_corr.columns:
                col_nome = "Corretor"
        if col_ger is None:
            if "IdGerente" in df_corr.columns:
                col_ger = "IdGerente"
            elif "Gerente" in df_corr.columns:
                col_ger = "Gerente"

        for _, r in df_corr.iterrows():
            idcor = str(r.get(col_idcor, "")).strip() if col_idcor else ""
            idimv = str(r.get(col_idimv, "")).strip() if col_idimv else ""
            nome = str(r.get(col_nome, "")).strip() if col_nome else ""
            ger = str(r.get(col_ger, "")).strip() if col_ger else ""

            if idcor and idimv and idimv.lower() != "nan":
                imoview_to_idcorretor[idimv] = idcor

            if idcor and nome and nome.lower() != "nan":
                nome_to_idcorretor[norm(nome)] = idcor

            if idcor and ger and ger.lower() != "nan":
                corretor_to_gerente[idcor] = ger

    return {
        "df_bairro": df_bairro,
        "df_tipo": df_tipo,
        "df_imovel": df_imovel,
        "df_fato": df_fato,
        "bairro_name_to_id": bairro_name_to_id,
        "tipo_name_to_id": tipo_name_to_id,
        "imoview_to_idcorretor": imoview_to_idcorretor,
        "nome_to_idcorretor": nome_to_idcorretor,
        "corretor_to_gerente": corretor_to_gerente,
    }


def next_dim_id(existing_ids: List[str], prefix: str) -> str:
    nums = []
    for x in existing_ids:
        if isinstance(x, str) and x.startswith(prefix):
            try:
                nums.append(int(x[len(prefix):]))
            except Exception:
                pass
    n = max(nums) + 1 if nums else 1
    return f"{prefix}{n}"


def ensure_bairro(dim: dict, bairro_nome: str) -> Tuple[str, bool]:
    key = norm(bairro_nome)
    if not key:
        return "", False

    if key in dim["bairro_name_to_id"]:
        return dim["bairro_name_to_id"][key], False

    df_bairro = dim["df_bairro"]
    if df_bairro.empty:
        df_bairro = pd.DataFrame(columns=["IdBairro", "Nome"])

    if "IdBairro" not in df_bairro.columns:
        df_bairro["IdBairro"] = ""
    if "Nome" not in df_bairro.columns:
        df_bairro["Nome"] = ""

    existing = df_bairro["IdBairro"].astype(str).tolist()
    new_id = next_dim_id(existing, "B")

    new_row = pd.DataFrame([{"IdBairro": new_id, "Nome": bairro_nome}])
    dim["df_bairro"] = pd.concat([df_bairro, new_row], ignore_index=True)
    dim["bairro_name_to_id"][key] = new_id
    return new_id, True


def map_tipo(dim: dict, tipo_nome: str) -> str:
    key = norm(tipo_nome)
    if not key:
        return ""
    return dim["tipo_name_to_id"].get(key, "")


def is_residencial_from_tipo_id(tipo_id: str) -> bool:
    # ajuste conforme seu Dim_Tipo (aqui é heurística)
    non_res = {"T7", "T8", "T9", "T10", "T11"}
    return str(tipo_id).strip().upper() not in non_res if tipo_id else True


# =========================
# UPSERT / APPEND NAS ABAS
# =========================
def upsert_dim_imovel(dim: dict, codigo: int, tipo_id: str, valor: float, bairro_id: str, foco_pp: bool, foco_ac: bool):
    df = dim["df_imovel"]
    if df.empty:
        df = pd.DataFrame(columns=["Código", "Tipo", "Valor", "Bairro", "Foco PP", "Foco AC"])

    for col in ["Código", "Tipo", "Valor", "Bairro", "Foco PP", "Foco AC"]:
        if col not in df.columns:
            df[col] = ""

    df["Código_num"] = pd.to_numeric(df["Código"], errors="coerce")
    mask = (df["Código_num"] == codigo)

    if mask.any():
        idx = df.index[mask][0]
        df.at[idx, "Tipo"] = tipo_id
        df.at[idx, "Valor"] = valor
        df.at[idx, "Bairro"] = bairro_id
        df.at[idx, "Foco PP"] = "TRUE" if foco_pp else "FALSE"
        df.at[idx, "Foco AC"] = "TRUE" if foco_ac else "FALSE"
    else:
        df = pd.concat([df, pd.DataFrame([{
            "Código": codigo,
            "Tipo": tipo_id,
            "Valor": valor,
            "Bairro": bairro_id,
            "Foco PP": "TRUE" if foco_pp else "FALSE",
            "Foco AC": "TRUE" if foco_ac else "FALSE",
        }])], ignore_index=True)

    df = df.drop(columns=["Código_num"], errors="ignore")
    dim["df_imovel"] = df


def append_fato_captacao(dim: dict, codigo: int, captadores: List[str], gerente: str, data_entrada: date):
    df = dim["df_fato"]
    if df.empty:
        df = pd.DataFrame(columns=["Código", "Captador1", "Captador2", "Captador3", "Gerente", "DataEntrada"])

    for col in ["Código", "Captador1", "Captador2", "Captador3", "Gerente", "DataEntrada"]:
        if col not in df.columns:
            df[col] = ""

    capt1 = captadores[0] if len(captadores) > 0 else ""
    capt2 = captadores[1] if len(captadores) > 1 else ""
    capt3 = captadores[2] if len(captadores) > 2 else ""

    # evita duplicar por (Código, DataEntrada, Captador1)
    df_tmp = df.copy()
    df_tmp["Código_num"] = pd.to_numeric(df_tmp["Código"], errors="coerce")
    df_tmp["DataEntrada_date"] = df_tmp["DataEntrada"].apply(parse_date_any)
    df_tmp["Captador1_str"] = df_tmp["Captador1"].astype(str)

    if ((df_tmp["Código_num"] == codigo) &
        (df_tmp["DataEntrada_date"] == data_entrada) &
        (df_tmp["Captador1_str"] == str(capt1))).any():
        dim["df_fato"] = df
        return

    df = pd.concat([df, pd.DataFrame([{
        "Código": codigo,
        "Captador1": capt1,
        "Captador2": capt2,
        "Captador3": capt3,
        "Gerente": gerente,
        "DataEntrada": data_entrada.strftime("%d/%m/%Y"),
    }])], ignore_index=True)

    dim["df_fato"] = df


# =========================
# INPUT CÓDIGOS
# =========================
def load_codes_from_xlsx(xlsx_path: str, sheet_name: str) -> List[int]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    col = None
    for c in df.columns:
        if norm(c) in {"CODIGO", "CÓDIGO"}:
            col = c
            break
    if col is None:
        raise RuntimeError("No XLSX, a aba indicada precisa ter uma coluna 'Código'.")

    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[col])
    return sorted(df[col].astype(int).unique().tolist())


# =========================
# MAIN
# =========================
def main():
    if not CREDENTIALS_JSON or not os.path.exists(CREDENTIALS_JSON):
        raise RuntimeError(f"Credenciais do Google não encontradas: {CREDENTIALS_JSON}")

    if not SPREADSHEET_ID or "COLE_AQUI" in SPREADSHEET_ID:
        raise RuntimeError("Defina SPREADSHEET_ID na seção CONFIG.")

    svc = sheets_service(CREDENTIALS_JSON)
    dim = load_dim_maps(svc, SPREADSHEET_ID)

    # 1) Carregar códigos
    codes: List[int] = []
    if CODES_XLSX_PATH:
        if not os.path.exists(CODES_XLSX_PATH):
            raise RuntimeError(f"XLSX não encontrado: {CODES_XLSX_PATH}")
        codes = load_codes_from_xlsx(CODES_XLSX_PATH, CODES_XLSX_SHEET)
    elif CODES_LIST:
        codes = sorted(set([int(x) for x in CODES_LIST]))
    else:
        # modo data
        if "DATE_FROM" in globals() and "DATE_TO" in globals():
            pass
        else:
            raise RuntimeError("Escolha um modo na CONFIG: CODES_LIST ou CODES_XLSX_PATH ou (DATE_FROM e DATE_TO).")

    # 2) Buscar no Imoview
    imoveis: Dict[int, dict] = {}

    if codes:
        imoveis = fetch_imoveis_by_codes(codes, FINALIDADE)
    else:
        # modo data
        if "DATE_FROM" not in globals() or "DATE_TO" not in globals():
            raise RuntimeError("Para modo data, defina DATE_FROM e DATE_TO na CONFIG.")
        imoveis = fetch_imoveis_by_date(globals()["DATE_FROM"], globals()["DATE_TO"], FINALIDADE)

    if not imoveis:
        print("Nada retornou do Imoview. Encerrando.")
        if codes:
            pd.DataFrame({"codigo_faltante": codes}).to_csv("codigos_nao_retornaram.csv", index=False, encoding="utf-8-sig")
            print(f"Arquivo gerado: codigos_nao_retornaram.csv (todos faltantes)")
        return

    # 3) Relatório de faltantes (quando você passou lista de códigos)
    if codes:
        codes_set = set(codes)
        returned_set = set(imoveis.keys())
        missing = sorted(codes_set - returned_set)
        returned_sorted = sorted(returned_set)

        print(f"Total planilha/lista: {len(codes)} | Retornados: {len(returned_set)} | Faltantes: {len(missing)}")
        if missing:
            print("Códigos que NÃO retornaram do endpoint (provavelmente não estão vago/disponível):")
            print(missing)

        pd.DataFrame({"codigo_retornado": returned_sorted}).to_csv("codigos_retornados.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"codigo_faltante": missing}).to_csv("codigos_nao_retornaram.csv", index=False, encoding="utf-8-sig")
        print("Arquivos gerados: codigos_retornados.csv | codigos_nao_retornaram.csv")

    # 4) Processar e atualizar Sheets
    novos_bairros_criados = False

    for codigo, item in imoveis.items():
        data = extract_fields(item)

        codigo = int(data["codigo"])
        bairro_nome = data["bairro_nome"]
        valor = float(data["valor"])
        tipo_nome = data["tipo_nome"]
        comissao = float(data["comissao_pct"])

        captadores_imoview_ids = data["captador_ids"]       # pode vir vazio dependendo do JSON
        captadores_imoview_nomes = data["captador_nomes"]   # fallback por nome

        # resolver captadores:
        # 1) se tiver id -> IdImoview -> IdCorretor
        # 2) se não tiver id -> tenta Nome -> IdCorretor
        # 3) se não der -> grava o que tiver (id ou nome) como fallback final
        captadores: List[str] = []

        if captadores_imoview_ids:
            for cid in captadores_imoview_ids:
                key = str(cid).strip()
                mapped = dim["imoview_to_idcorretor"].get(key, "")
                captadores.append(mapped if mapped else key)
        elif captadores_imoview_nomes:
            for nome in captadores_imoview_nomes:
                mapped = dim["nome_to_idcorretor"].get(norm(nome), "")
                captadores.append(mapped if mapped else nome)
        else:
            captadores = []

        captadores = [str(x).strip() for x in captadores if str(x).strip()][:3]

        data_entrada = data["data_entrada"] or date.today()

        bairro_id, created = ensure_bairro(dim, bairro_nome)
        if created:
            novos_bairros_criados = True

        tipo_id = map_tipo(dim, tipo_nome) or ""
        residencial = is_residencial_from_tipo_id(tipo_id)

        foco_pp, foco_ac = classificar_foco(bairro_nome, valor, comissao, residencial)

        # gerente: tenta por capt1 (se for IdCorretor tipo C61xxx)
        capt1 = captadores[0] if captadores else ""
        gerente = dim["corretor_to_gerente"].get(str(capt1), "")

        upsert_dim_imovel(dim, codigo, tipo_id, valor, bairro_id, foco_pp, foco_ac)
        append_fato_captacao(dim, codigo, captadores, gerente, data_entrada)

        print(
            f"[OK] {codigo} | {bairro_nome} | valor={valor:.0f} | com={comissao} | "
            f"capt(imoview_ids)={captadores_imoview_ids} capt(imoview_nomes)={captadores_imoview_nomes} -> "
            f"capt(final)={captadores} | ger={gerente} | PP={foco_pp} AC={foco_ac}"
        )

    # 5) Persistir no Google Sheets
    if novos_bairros_criados:
        write_df_over_sheet(svc, SPREADSHEET_ID, "Dim_Bairro", dim["df_bairro"])

    write_df_over_sheet(svc, SPREADSHEET_ID, "Dim_Imovel", dim["df_imovel"])
    write_df_over_sheet(svc, SPREADSHEET_ID, "Fato_Captacao", dim["df_fato"])

    print("\nFinalizado: Dim_Imovel e Fato_Captacao atualizados no Google Sheet.")


if __name__ == "__main__":
    main()
