# =========================
# tratamento_estoque_imoview_robusto.py
# - Lê export do Imoview mesmo quando vem ".xls" mas na verdade é HTML
# - Monta Endereco: Endereco + EnderecoNumero + Bloco + Complemento
# - Cria Captadores_final a partir de Captadores ou Captador1/2/3
# - Exporta Excel no padrão final
# =========================

import os
import pandas as pd
import numpy as np

INPUT_FILE = "./imoveis-2026-02-06-014845.xls"
OUTPUT_XLSX = "./Estoque 2026-02-07.xlsx"


def sniff_is_html(path: str) -> bool:
    """Detecta se o arquivo é HTML disfarçado (começa com <div, <html etc)."""
    with open(path, "rb") as f:
        head = f.read(2048).lstrip()
    head_low = head.lower()
    return head_low.startswith(b"<") and (
        b"<html" in head_low or b"<div" in head_low or b"<table" in head_low
    )


def ler_arquivo_robusto(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    # Se for xlsx/xlsm -> openpyxl
    if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        return pd.read_excel(path, engine="openpyxl")

    # Se for "xls" mas na verdade html, lê como tabela html
    if ext == ".xls" and sniff_is_html(path):
        tables = pd.read_html(path, flavor="lxml")  # retorna lista de dataframes
        if not tables:
            raise ValueError("Não encontrei nenhuma tabela HTML dentro do arquivo.")
        # escolhe a maior tabela (normalmente é a do estoque)
        return max(tables, key=lambda t: t.shape[0] * t.shape[1])

    # Se for .xls real: tenta xlrd (se existir)
    if ext == ".xls":
        try:
            return pd.read_excel(path, engine="xlrd")
        except Exception as e:
            # última tentativa: se não for html e xlrd falhou, mostra mensagem clara
            raise RuntimeError(
                "Falha ao ler .xls. Se o arquivo veio do sistema, pode ser HTML disfarçado.\n"
                f"Erro original: {e}"
            )

    # fallback genérico
    return pd.read_excel(path)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def montar_endereco(row: pd.Series, col_end, col_num, col_bloco, col_comp) -> str:
    end = clean_str(row.get(col_end)) if col_end else ""
    num = clean_str(row.get(col_num)) if col_num else ""
    bloco = clean_str(row.get(col_bloco)) if col_bloco else ""
    comp = clean_str(row.get(col_comp)) if col_comp else ""

    partes = []
    if end:
        partes.append(end)
    if num:
        partes.append(num)

    # só coloca bloco se for algo real
    if bloco and bloco.lower() not in {"n", "na", "sn", "s/n"}:
        partes.append(f"Bloco {bloco}")

    if comp:
        partes.append(comp)

    return " ".join(partes).strip()


def get_col_or_nd(df: pd.DataFrame, colname: str | None):
    if colname and colname in df.columns:
        return df[colname]
    return "N/D"


# =========================
# EXECUÇÃO
# =========================
df = ler_arquivo_robusto(INPUT_FILE)

# Mapeamento tolerante
col_codigo = first_existing(df, ["Codigo", "Código", "CODIGO", "CÓDIGO"])
if not col_codigo:
    raise ValueError("Não encontrei a coluna de Código (Codigo/Código).")

col_captadores = first_existing(df, ["Captadores", "CAPTADORES"])
cap1 = first_existing(df, ["Captador1", "CAPTADOR1", "Captador_1"])
cap2 = first_existing(df, ["Captador2", "CAPTADOR2", "Captador_2"])
cap3 = first_existing(df, ["Captador3", "CAPTADOR3", "Captador_3"])

col_tipo = first_existing(df, ["Tipo", "TIPO", "Categoria", "CATEGORIA"])
col_quartos = first_existing(df, ["NumeroQuarto", "NúmeroQuarto", "NUMEROQUARTO", "Quartos", "Dormitorios", "DORMITORIOS"])
col_valor = first_existing(df, ["Valor", "VALOR", "Preco", "Preço", "PRECO", "PREÇO"])
col_bairro = first_existing(df, ["Bairro", "BAIRRO"])
col_pub = first_existing(df, ["PublicacaoNaInternet", "PublicaçãoNaInternet", "PUBLICACAONAINTERNET"])
col_exc = first_existing(df, ["Exclusivo", "EXCLUSIVO"])
col_lote = first_existing(df, ["AreaLote", "ÁreaLote", "AREALOTE", "Area do Lote"])
col_int = first_existing(df, ["AreaInterna", "ÁreaInterna", "AREAINTERNA", "Area Interna"])

col_end = first_existing(df, ["Endereco", "Endereço", "ENDERECO", "ENDEREÇO"])
col_num = first_existing(df, ["EnderecoNumero", "EndereçoNumero", "ENDERECONUMERO", "Numero", "Número", "NUMERO"])
col_bloco = first_existing(df, ["Bloco", "BLOCO"])
col_comp = first_existing(df, ["Complemento", "COMPLEMENTO"])

# Captadores_final
if col_captadores:
    df["Captadores_final"] = df[col_captadores].apply(clean_str)
else:
    cols_caps = [c for c in [cap1, cap2, cap3] if c]
    if cols_caps:
        def juntar_caps(row):
            vals = [clean_str(row.get(c)) for c in cols_caps]
            vals = [v for v in vals if v]
            return " | ".join(vals) if vals else "N/D"
        df["Captadores_final"] = df.apply(juntar_caps, axis=1)
    else:
        df["Captadores_final"] = "N/D"

# Endereco_final
df["Endereco_final"] = df.apply(lambda r: montar_endereco(r, col_end, col_num, col_bloco, col_comp), axis=1)
df.loc[df["Endereco_final"].eq(""), "Endereco_final"] = "N/D"

# Saída final padronizada
out = pd.DataFrame({
    "Codigo": df[col_codigo],
    "Captadores": df["Captadores_final"],
    "Tipo": get_col_or_nd(df, col_tipo),
    "NumeroQuarto": get_col_or_nd(df, col_quartos),
    "Valor": get_col_or_nd(df, col_valor),
    "Endereco": df["Endereco_final"],
    "Bairro": get_col_or_nd(df, col_bairro),
    "PublicacaoNaInternet": get_col_or_nd(df, col_pub),
    "Exclusivo": get_col_or_nd(df, col_exc),
    "AreaLote": get_col_or_nd(df, col_lote),
    "AreaInterna": get_col_or_nd(df, col_int),
})

out.to_excel(OUTPUT_XLSX, index=False)

print("OK! Gerado:", OUTPUT_XLSX)
print("Colunas lidas:", list(df.columns))
print(out.head(10))
