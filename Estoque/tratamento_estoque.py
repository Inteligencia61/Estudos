import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# CONFIG
# =========================
INPUT_FILE = "./imoveis-2026-08-07.xlsx"  # Altere para o caminho do seu arquivo (.xlsx ou .csv)
OUTPUT_XLSX = "./Fato_Estoque_IMPORTAR.xlsx"  # Arquivo pronto para o Apps Script

# Data do estoque como data real (célula de Excel)
DATA_ESTOQUE = datetime.now().date()

# =========================
# HELPERS
# =========================
def clean_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    # Se for número inteiro float vindo do pandas (ex: 12320.0), converte para string sem decimal
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def split_captadores(valor: str):
    """
    No Imoview, 'Captadores' costuma vir assim:
      "Nome 1 | Nome 2 | Nome 3"
    Gera Captador1..3 (nomes).
    """
    s = clean_str(valor)
    if not s:
        return "", "", ""
    parts = [p.strip() for p in s.split("|")]
    parts = [p for p in parts if p]
    parts = (parts + ["", "", ""])[:3]
    return parts[0], parts[1], parts[2]

# =========================
# LEITURA DO ARQUIVO (.XLSX / .CSV / .XLS)
# =========================
print(f"Lendo arquivo: {INPUT_FILE}...")

if INPUT_FILE.lower().endswith(".csv"):
    # Tenta ler CSV testando separadores comuns (; e ,)
    try:
        df = pd.read_csv(INPUT_FILE, sep=";", dtype=str, encoding="utf-8-sig")
        if len(df.columns) <= 1:
            df = pd.read_csv(INPUT_FILE, sep=",", dtype=str, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(INPUT_FILE, sep=",", dtype=str, encoding="utf-8-sig")
else:
    # Tenta ler como Excel XLSX/XLS nativo
    try:
        df = pd.read_excel(INPUT_FILE, dtype=str)
    except Exception:
        # Fallback para HTML disfarçado de XLS (se for o export clássico do Imoview)
        tables = pd.read_html(INPUT_FILE, flavor="lxml")
        if not tables:
            raise ValueError("Não foi possível ler o arquivo fornecido.")
        df = max(tables, key=lambda t: t.shape[0] * t.shape[1]).copy()

# =========================
# VALIDAR COLUNAS DO EXPORT
# =========================
cols_lower = {str(col).strip().lower(): col for col in df.columns}

if "codigo" in cols_lower:
    col_codigo = cols_lower["codigo"]
elif "código" in cols_lower:
    col_codigo = cols_lower["código"]
else:
    raise ValueError(f"Não encontrei a coluna 'Codigo' ou 'Código'. Colunas encontradas: {list(df.columns)}")

if "captadores" in cols_lower:
    col_captadores = cols_lower["captadores"]
else:
    df["Captadores"] = ""
    col_captadores = "Captadores"

# =========================
# GERAR Captador1..3 (NOMES)
# =========================
caps = df[col_captadores].apply(split_captadores)
df["Captador1"] = caps.apply(lambda x: x[0])
df["Captador2"] = caps.apply(lambda x: x[1])
df["Captador3"] = caps.apply(lambda x: x[2])

# =========================
# SAÍDA EXATA PARA processFile(data)
# (6 COLUNAS: A=Codigo, B=Captador1, C=Captador2, D=Captador3, E=Gerente, F=DataEstoque)
# =========================
out = pd.DataFrame({
    "Codigo": df[col_codigo].apply(clean_str),    # A
    "Captador1": df["Captador1"],                 # B
    "Captador2": df["Captador2"],                 # C
    "Captador3": df["Captador3"],                 # D
    "Gerente": "",                                # E (vazio para o Apps Script preencher)
    "DataEstoque": pd.to_datetime(DATA_ESTOQUE),  # F (Data nativa de Excel)
})

# Remover linhas sem código
out = out[out["Codigo"].astype(str).str.strip().ne("")].copy()

# =========================
# SALVAR XLSX COM FORMATO DE DATA
# =========================
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", date_format="YYYY-MM-DD", datetime_format="YYYY-MM-DD") as writer:
    out.to_excel(writer, index=False)

print("OK! XLSX compatível gerado com sucesso:", OUTPUT_XLSX)
print(out.head(10))