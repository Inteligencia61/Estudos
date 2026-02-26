import pandas as pd
import numpy as np
import glob
import os
import re

# =====================================================
# CONFIG
# =====================================================

CONFIG = {
    "pattern": "../2025-11*.csv ../2025-12*.csv ../2026-01*.csv",
    "PRECO_MIN": 300_000,
    "PRECO_MAX": 50_000_000,
    "APLICAR_IQR": True
}


# =====================================================
# UTILIDADES
# =====================================================

def achar_coluna_valor(df):
    for c in df.columns:
        if "valor" in str(c).lower() or "preco" in str(c).lower():
            return c
    raise Exception("Coluna de valor não encontrada")


def extrair_data(path):
    nome = os.path.basename(path)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", nome)
    if m:
        return pd.to_datetime(m.group(0))
    return None


def parse_numero(x):
    if pd.isna(x):
        return np.nan
    
    s = str(x)
    s = re.sub(r"[^\d,\.]", "", s)
    
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    
    return pd.to_numeric(s, errors="coerce")


def corrigir_unidade(series):
    s = series.dropna()
    if s.empty:
        return series
    
    med = s.median()
    
    # Detecta escala
    if med < 1000:
        fator = 1_000_000
    elif med < 200_000:
        fator = 1_000
    else:
        fator = 1
    
    return series * fator


def remover_outliers_iqr(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    
    return df[(df[col] >= low) & (df[col] <= high)]


def resumo(s):
    return {
        "qtd": len(s),
        "media": s.mean(),
        "mediana": s.median(),
        "p10": s.quantile(0.10),
        "p90": s.quantile(0.90)
    }


def fmt(x):
    if pd.isna(x):
        return "-"
    return f"{x:,.0f}".replace(",", ".")


# =====================================================
# 1. CARREGAR TODOS OS ARQUIVOS
# =====================================================

arquivos = []
for p in CONFIG["pattern"].split():
    arquivos += glob.glob(p)

dados = []

print("\n===== LENDO ARQUIVOS =====\n")

for arq in arquivos:
    df = pd.read_csv(arq, low_memory=False)
    soma = 0
    for index, a in enumerate(df['preco']):
        soma += a
    print(soma)
    print(soma / index)
    
    data = extrair_data(arq)
    if data is None:
        continue
    
    col = achar_coluna_valor(df)
    
    valores = df[col].apply(parse_numero)
    valores = corrigir_unidade(valores)
    
    # Limites de mercado
    valores = valores.where(
        (valores >= CONFIG["PRECO_MIN"]) &
        (valores <= CONFIG["PRECO_MAX"])
    )
    
    temp = pd.DataFrame({
        "data": data,
        "valor": valores
    }).dropna()
    
    if len(temp) < 1000:
        print(f"Arquivo ignorado (amostra baixa): {arq}")
        continue
    
    dados.append(temp)

df_total = pd.concat(dados, ignore_index=True)

# =====================================================
# 2. OUTLIERS
# =====================================================

if CONFIG["APLICAR_IQR"]:
    df_total = remover_outliers_iqr(df_total, "valor")

# =====================================================
# 3. CRIAR MÊS
# =====================================================

df_total["mes"] = df_total["data"].dt.to_period("M")

# =====================================================
# 4. RESUMO POR MÊS
# =====================================================

print("\n===== TENDÊNCIA MENSAL =====\n")

resultado = []

for mes, grupo in df_total.groupby("mes"):
    r = resumo(grupo["valor"])
    resultado.append({
        "mes": str(mes),
        **r
    })

df_res = pd.DataFrame(resultado).sort_values("mes")

for _, row in df_res.iterrows():
    print(f"\nMÊS: {row['mes']}")
    print(f"Qtd: {int(row['qtd'])}")
    print(f"Mediana: {fmt(row['mediana'])}")
    print(f"Média:   {fmt(row['media'])}")
    print(f"P10/P90: {fmt(row['p10'])} / {fmt(row['p90'])}")

# =====================================================
# 5. VARIAÇÃO ENTRE MESES
# =====================================================

print("\n===== VARIAÇÃO MENSAL =====\n")

df_res["var_mediana_%"] = df_res["mediana"].pct_change() * 100

for _, row in df_res.iterrows():
    if pd.isna(row["var_mediana_%"]):
        continue
    
    print(f"{row['mes']}: variação mediana = {row['var_mediana_%']:.1f}%")

# Exportar
df_res.to_csv("tendencia_mensal_lago_norte.csv", index=False, encoding="utf-8-sig")

print("\nArquivo gerado: tendencia_mensal_lago_norte.csv")