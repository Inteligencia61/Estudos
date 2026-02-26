import csv
import os

def transferir_csv():
    arquivo_origem = '/Users/imac/Downloads/2024-10-07 (Estudos) - 2024-10-07 (1).csv.csv'
    arquivo_destino = '/Users/imac/Desktop/Esquema Inteligencia/Cod_ETL_estudos/Base_estudos.csv'
    
    # Define as colunas desejadas
    colunas_desejadas = [
        "codigo", "anunciante", "oferta", "tipo",
        "area_util", "bairro", "cidade", "preco", "valor_m2",
        "quartos", "vagas", "data", "quadra"
    ]
    
    # Lê e trata os dados da planilha de origem
    with open(arquivo_origem, 'r') as f_origem:
        leitor_origem = csv.DictReader(f_origem)
        
        # Filtra as linhas para garantir que apenas as com as colunas desejadas sejam mantidas
        dados_tratados = [
            {coluna: str(linha[coluna]) for coluna in colunas_desejadas}
            for linha in leitor_origem
            if set(linha.keys()) == set(colunas_desejadas)
        ]
    
    # Verificar o número de linhas já presentes no arquivo de destino
    if os.path.exists(arquivo_destino):
        with open(arquivo_destino, 'r') as f_destino:
            num_linhas_destino = sum(1 for _ in csv.reader(f_destino))
    else:
        num_linhas_destino = 0
    
    # Define o número máximo de linhas a serem transferidas
    maximo_linhas = min(50000, len(dados_tratados))
    
    # Adiciona os dados tratados ao arquivo de destino
    with open(arquivo_destino, 'a', newline='') as f_destino:
        escritor_destino = csv.DictWriter(f_destino, fieldnames=colunas_desejadas)
        
        # Escrever o cabeçalho apenas se o arquivo estiver vazio
        if num_linhas_destino == 0:
            escritor_destino.writeheader()
        
        escritor_destino.writerows(dados_tratados[:maximo_linhas])
    
    print(f"A transferência de {maximo_linhas} linhas foi concluída com sucesso! O arquivo destino agora tem {num_linhas_destino + maximo_linhas} linhas.")

# Chama a função para fazer a transferência
if __name__ == '__main__':
    transferir_csv()
