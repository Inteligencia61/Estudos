import pandas as pd

# Função para filtrar a base de dados
def filtrar_por_nome(nome):
    # Carregar a base de dados (ajuste o caminho conforme necessário)
    df = pd.read_csv('./Base Inteligência 61 - Fato_Venda (3).csv')
    
    # Corrigir nomes das colunas, removendo espaços extras
    df.columns = df.columns.str.strip()

    # Filtrar as linhas onde o nome aparece em qualquer uma das colunas
    resultado = df[df[['vendedor 1', 'Vendedor 2', 'Captador 1', 'Captador 2']].apply(lambda x: x.str.contains(nome, case=False, na=False)).any(axis=1)]
    
    return resultado

# Exemplo de uso
nome = input("Digite o nome a ser filtrado: ")
resultado = filtrar_por_nome(nome)

# Exibir o resultado
print(f"Linhas encontradas com o nome '{nome}':")
print(resultado)
resultado.to_csv("pavoni_filter.csv")