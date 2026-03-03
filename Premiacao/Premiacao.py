# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

# Tenta importar gspread, mas permite execução local sem ele
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    gspread = None
    ServiceAccountCredentials = None

# ==========================================================
# CONFIGURAÇÕES DO RELATÓRIO
# ==========================================================
MES_RELATORIO = 2  # Fevereiro
ANO_RELATORIO = 2026
GERENTES_ESPECIAIS = ['HELIO', 'LUANA']
EXCECAO_OUTROS = ['FERNANDA LINDSAY']

# ==========================================================
# FUNÇÕES DE APOIO (BASEADAS NO SEU EXEMPLO)
# ==========================================================
def _to_float_br(x) -> float:
    """
    Função de conversão idêntica ao seu exemplo que funcionou.
    """
    #print(x)
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return 0.0
    s = s.replace("R$", "").strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    #print(x)
    # Mantém apenas números, pontos e sinais de menos
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    
    try:
        return float(s)
    except:
        return 0.0

def limpar_nome(n):
    if pd.isna(n) or str(n).strip() in ["", "-", "nan", "NAN", "None"]: 
        return ""
    return str(n).strip().upper()

def autenticar_google_sheets():
    if not gspread:
        raise ImportError("Biblioteca gspread não instalada.")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("../cred.json", scope)
    client = gspread.authorize(creds)
    return client

def carregar_dados_sheets():
    client = autenticar_google_sheets()
    sheet_id = "1SyyKK8dbL1-PKL364eA2B34Vxp6QJhbZ9qv94mFLmLI"

    planilha = client.open_by_key(sheet_id)
    aba = planilha.get_worksheet(0)  # primeira aba

    dados = aba.get_all_values()

    # Primeira linha = cabeçalho
    header = dados[0]
    rows = dados[1:]

    df = pd.DataFrame(rows, columns=header)

    # Remove espaços nos nomes das colunas
    df.columns = df.columns.str.strip()

    # Remove linhas completamente vazias
    df = df.dropna(how="all")

    return df

# ==========================================================
# LÓGICA DE PROCESSAMENTO DOS RANKINGS
# ==========================================================
def processar_rankings(df_subset, tipo_entidade='CORRETOR'):
    vgv_cap, vgv_vend, vgv_geral = {}, {}, {}
    vgc_cap, vgc_vend, vgc_geral = {}, {}, {}

    for _, row in df_subset.iterrows():
        v_imovel = row.get('Valor_Negocio', 0) or 0
        v_comissao_total = row.get('Valor_Total_61', 0) or 0

        # --- nomes por papel ---
        if tipo_entidade == 'CORRETOR':
            v_nomes = {
                limpar_nome(row.get('Corretor_Venda_1_Nome')),
                limpar_nome(row.get('Corretor_Venda_2_Nome')),
            }
            c_nomes = {
                limpar_nome(row.get('Corretor_Captador_1_Nome')),
                limpar_nome(row.get('Corretor_Captador_2_Nome')),
            }
        else:
            v_nomes = {limpar_nome(row.get('Gerente_Venda_Nome'))}
            c_nomes = {limpar_nome(row.get('Gerente_Captacao_Nome'))}

        v_nomes = {n for n in v_nomes if n}
        c_nomes = {n for n in c_nomes if n}
        todos_nomes = v_nomes | c_nomes

        # --- RANKINGS VGV (mantém sua regra atual) ---
        for n in v_nomes:
            vgv_vend[n] = vgv_vend.get(n, 0) + v_imovel
        for n in c_nomes:
            vgv_cap[n] = vgv_cap.get(n, 0) + v_imovel
        for n in todos_nomes:
            vgv_geral[n] = vgv_geral.get(n, 0) + v_imovel

        # --- RANKINGS VGC (corrigido: divide por lado e por pessoa) ---

        tem_venda = len(v_nomes) > 0
        tem_capt = len(c_nomes) > 0

        if tem_venda and tem_capt:
            parcela_venda = v_comissao_total * 0.5
            parcela_capt = v_comissao_total * 0.5
        elif tem_venda and not tem_capt:
            parcela_venda = v_comissao_total
            parcela_capt = 0.0
        elif tem_capt and not tem_venda:
            parcela_venda = 0.0
            parcela_capt = v_comissao_total
        else:
            parcela_venda = 0.0
            parcela_capt = 0.0

        # divide igualmente entre as pessoas de cada lado
        if tem_venda and parcela_venda:
            por_vendedor = parcela_venda / len(v_nomes)
            for n in v_nomes:
                vgc_vend[n] = vgc_vend.get(n, 0) + por_vendedor

        if tem_capt and parcela_capt:
            por_captador = parcela_capt / len(c_nomes)
            for n in c_nomes:
                vgc_cap[n] = vgc_cap.get(n, 0) + por_captador

        # geral: soma exatamente o que cada um recebeu (sem duplicar total)
        # (se alguém aparecer nos dois lados na mesma linha, ele acumula as duas parcelas)
        for n in v_nomes:
            # pode não ter sido adicionado se parcela_venda=0
            if tem_venda and parcela_venda:
                vgc_geral[n] = vgc_geral.get(n, 0) + (parcela_venda / len(v_nomes))
        for n in c_nomes:
            if tem_capt and parcela_capt:
                vgc_geral[n] = vgc_geral.get(n, 0) + (parcela_capt / len(c_nomes))

    return {
        'VGV_CAP': vgv_cap, 'VGV_VEND': vgv_vend, 'VGV_GERAL': vgv_geral,
        'VGC_CAP': vgc_cap, 'VGC_VEND': vgc_vend, 'VGC_GERAL': vgc_geral
    }

def imprimir_ranking(titulo, dados):
    print(f"\n--- {titulo} ---")
    if not dados:
        print("Sem dados.")
        return
    sorted_dados = sorted(dados.items(), key=lambda x: x[1], reverse=True)
    for n, v in sorted_dados:
        print(f"{n.ljust(30)} | R$ {v:14,.2f}")

# ==========================================================
# EXECUÇÃO
# ==========================================================
def main(file_path=None):
    if file_path:
        df = pd.read_excel(file_path, sheet_name="Vendas")
    else:
        try:
            df = carregar_dados_sheets()
        except Exception as e:
            print(f"Erro ao carregar planilha: {e}")
            return

    #df.columns = df.columns.str.strip()

    # 1. Limpeza Financeira (Usando a função do seu exemplo)
    #print(df['Valor_Negocio'])
    #print(df['Valor_Total_61'])
    cols_fin = ['Valor_Negocio', 'Valor_Total_61']
    for col in cols_fin:
        if col in df.columns:
            df[col] = df[col].apply(_to_float_br)

    # 2. Filtro de Data
    df['Data_Contrato'] = pd.to_datetime(df['Data_Contrato'], errors='coerce')
    #df_mes = df[(df['Data_Contrato'].dt.month == MES_RELATORIO) & (df['Data_Contrato'].dt.year == ANO_RELATORIO)].copy()
    df_mes = df[(df['Data_Contrato'] >= '2026-02-01') & (df['Data_Contrato'] <= '2026-02-28')]
    #print(len(df_mes))
    #print(df_mes['Valor_Negocio'])
    som = 0
    for a in df_mes['Valor_Negocio']:
        som += a
    #print(som)
    # 3. Separação de Equipes
    def is_especial(row):
        gv = limpar_nome(row.get('Gerente_Venda_Nome'))
        gc = limpar_nome(row.get('Gerente_Captacao_Nome'))
        return any(g in gv or g in gc for g in GERENTES_ESPECIAIS)

    mask_especial = df_mes.apply(is_especial, axis=1)
    df_esp_raw = df_mes[mask_especial].copy()
    df_outros_raw = df_mes[~mask_especial].copy()

    # 4. Processamento dos Grupos
    res_esp = processar_rankings(df_esp_raw, 'CORRETOR')
    res_outros = processar_rankings(df_outros_raw, 'CORRETOR')
    res_gerentes = processar_rankings(df_mes, 'GERENTE')

    # 5. Ajuste Fernanda Lindsay
    for rank_tipo in ['VGV_CAP', 'VGV_VEND', 'VGV_GERAL', 'VGC_CAP', 'VGC_VEND', 'VGC_GERAL']:
        for nome in EXCECAO_OUTROS:
            if nome in res_esp[rank_tipo]:
                valor = res_esp[rank_tipo].pop(nome)
                res_outros[rank_tipo][nome] = res_outros[rank_tipo].get(nome, 0) + valor

    # ==========================================================
    # EXIBIÇÃO DOS RANKINGS
    # ==========================================================
    print(f"\n{'#'*60}")
    print(f" RELATÓRIO DE RANKINGS - {MES_RELATORIO}/{ANO_RELATORIO}")
    print(f"{'#'*60}")

    print("\n>>> EQUIPE HELIO / LUANA (VGV)")
    imprimir_ranking("VGV CAPTAÇÃO - AC", res_esp['VGV_CAP'])
    imprimir_ranking("VGV VENDA - AC", res_esp['VGV_VEND'])
    imprimir_ranking("VGV GERAL - AC", res_esp['VGV_GERAL'])

    print("\n>>> EQUIPE OUTROS (VGV)")
    imprimir_ranking("VGV CAPTAÇÃO - PP", res_outros['VGV_CAP'])
    imprimir_ranking("VGV VENDA - PP", res_outros['VGV_VEND'])
    imprimir_ranking("VGV GERAL - PP", res_outros['VGV_GERAL'])

    print("\n>>> EQUIPE HELIO / LUANA (VGC)")
    imprimir_ranking("VGC CAPTAÇÃO - AC", res_esp['VGC_CAP'])
    imprimir_ranking("VGC VENDA - AC", res_esp['VGC_VEND'])
    imprimir_ranking("VGC GERAL - AC", res_esp['VGC_GERAL'])

    print("\n>>> EQUIPE OUTROS (VGC)")
    imprimir_ranking("VGC CAPTAÇÃO - PP", res_outros['VGC_CAP'])
    imprimir_ranking("VGC VENDA - PP", res_outros['VGC_VEND'])
    imprimir_ranking("VGC GERAL - PP", res_outros['VGC_GERAL'])

    print("\n>>> RANKINGS GERENTES")
    imprimir_ranking("VGV GERAL - GERENTES", res_gerentes['VGV_GERAL'])
    imprimir_ranking("VGC GERAL - GERENTES", res_gerentes['VGC_GERAL'])
    imprimir_ranking("VGC CAPTAÇÃO - GERENTES", res_gerentes['VGC_CAP'])
    imprimir_ranking("VGC VENDA - GERENTES", res_gerentes['VGC_VEND'])

if __name__ == "__main__":
    # main("/home/ubuntu/upload/CópiadeControledeContratos61Imóveis.xlsx")
    main()
