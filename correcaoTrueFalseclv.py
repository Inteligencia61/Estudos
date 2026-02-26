import pandas as pd

# Arquivo original
arquivo_entrada = "Dim_Imovel.csv"

# Arquivo corrigido
arquivo_saida = "Dim_Imovel_corrigido.csv"

# Ler tudo como texto
df = pd.read_csv(arquivo_entrada, dtype=str)

# Colunas que podem ter TRUE/FALSE
colunas_bool = ["Foco PP", "Foco AC"]

for col in colunas_bool:
    if col in df.columns:
        df[col] = (
            df[col]
            .str.replace("'", "", regex=False)   # remove o '
            .str.upper()
        )

# Salvar CSV sem index
df.to_csv(arquivo_saida, index=False, encoding="utf-8-sig")

print("CSV corrigido gerado:", arquivo_saida)
