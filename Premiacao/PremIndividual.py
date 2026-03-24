# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from datetime import datetime

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    gspread = None
    ServiceAccountCredentials = None

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

MES_RELATORIO = 2
ANO_RELATORIO = 2026
TOP_N = 30

GERENTES_ESPECIAIS = ['HELIO', 'LUANA']
EXCECAO_OUTROS = ['FERNANDA LINDSAY']

# ==========================================================
# FUNÇÕES BASE
# ==========================================================

def _to_float_br(x):

    if x is None:
        return 0.0

    if isinstance(x,(int,float)):
        return float(x)

    s=str(x).strip()

    if not s:
        return 0.0

    s=s.replace("R$","").strip()

    if "," in s and "." in s:
        s=s.replace(".","").replace(",",".")
    elif "," in s:
        s=s.replace(",", ".")

    s="".join(ch for ch in s if ch.isdigit() or ch in ".-")

    try:
        return float(s)
    except:
        return 0.0


def limpar_nome(n):

    if pd.isna(n) or str(n).strip() in ["","-","nan","NAN","None"]:
        return ""

    return str(n).strip().upper()


# ==========================================================
# GOOGLE SHEETS
# ==========================================================

def autenticar_google_sheets():

    if not gspread:
        raise ImportError("gspread não instalado")

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_name("../cred.json", scope)

    client = gspread.authorize(creds)

    return client


def carregar_dados_sheets():

    client = autenticar_google_sheets()

    sheet_id = "1GLYIVuOG0heAXKxL5MdtjNxlR7o9N8BaWuvwHF9Jb0Y"

    planilha = client.open_by_key(sheet_id)

    aba = planilha.get_worksheet(0)

    dados = aba.get_all_values()

    header = dados[0]

    rows = dados[1:]

    df = pd.DataFrame(rows,columns=header)

    df.columns=df.columns.str.strip()

    df=df.dropna(how="all")

    return df


# ==========================================================
# RANKINGS (SEU CÓDIGO)
# ==========================================================

def processar_rankings(df_subset, tipo_entidade='CORRETOR'):

    vgv_cap, vgv_vend, vgv_geral = {}, {}, {}
    vgc_cap, vgc_vend, vgc_geral = {}, {}, {}

    for _, row in df_subset.iterrows():

        v_imovel = row.get('Valor_Negocio',0) or 0
        v_comissao_total = row.get('Valor_Total_61',0) or 0

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

            v_nomes={limpar_nome(row.get('Gerente_Venda_Nome'))}
            c_nomes={limpar_nome(row.get('Gerente_Captacao_Nome'))}

        v_nomes={n for n in v_nomes if n}
        c_nomes={n for n in c_nomes if n}

        todos=v_nomes|c_nomes

        for n in v_nomes:
            vgv_vend[n]=vgv_vend.get(n,0)+v_imovel

        for n in c_nomes:
            vgv_cap[n]=vgv_cap.get(n,0)+v_imovel

        for n in todos:
            vgv_geral[n]=vgv_geral.get(n,0)+v_imovel


        tem_venda=len(v_nomes)>0
        tem_capt=len(c_nomes)>0

        if tem_venda and tem_capt:
            parcela_venda=v_comissao_total*0.5
            parcela_capt=v_comissao_total*0.5

        elif tem_venda:
            parcela_venda=v_comissao_total
            parcela_capt=0

        elif tem_capt:
            parcela_venda=0
            parcela_capt=v_comissao_total

        else:
            parcela_venda=parcela_capt=0


        if tem_venda and parcela_venda:

            por_vendedor=parcela_venda/len(v_nomes)

            for n in v_nomes:

                vgc_vend[n]=vgc_vend.get(n,0)+por_vendedor
                vgc_geral[n]=vgc_geral.get(n,0)+por_vendedor


        if tem_capt and parcela_capt:

            por_capt=parcela_capt/len(c_nomes)

            for n in c_nomes:

                vgc_cap[n]=vgc_cap.get(n,0)+por_capt
                vgc_geral[n]=vgc_geral.get(n,0)+por_capt


    return {
        'VGV_CAP':vgv_cap,
        'VGV_VEND':vgv_vend,
        'VGV_GERAL':vgv_geral,
        'VGC_CAP':vgc_cap,
        'VGC_VEND':vgc_vend,
        'VGC_GERAL':vgc_geral
    }


# ==========================================================
# NOVO — IMÓVEIS POR CORRETOR
# ==========================================================

def extrair_imoveis_por_corretor(df):

    corretores = {}

    for _, row in df.iterrows():

        valor_imovel = row.get("Valor_Negocio", 0) or 0
        valor_comissao = row.get("Valor_Total_61", 0) or 0

        id_contrato = str(row.get("Contrato", "")).strip()

        if not id_contrato:
            id_contrato = f"Contrato_{_}"

        vendedores = {
            limpar_nome(row.get('Corretor_Venda_1_Nome')),
            limpar_nome(row.get('Corretor_Venda_2_Nome'))
        }

        captadores = {
            limpar_nome(row.get('Corretor_Captador_1_Nome')),
            limpar_nome(row.get('Corretor_Captador_2_Nome'))
        }

        vendedores = {n for n in vendedores if n}
        captadores = {n for n in captadores if n}

        todos = vendedores | captadores

        if not todos:
            continue

        tem_venda = len(vendedores) > 0
        tem_capt = len(captadores) > 0

        if tem_venda and tem_capt:
            parcela_venda = valor_comissao * 0.5
            parcela_capt = valor_comissao * 0.5
        elif tem_venda:
            parcela_venda = valor_comissao
            parcela_capt = 0
        elif tem_capt:
            parcela_venda = 0
            parcela_capt = valor_comissao
        else:
            parcela_venda = parcela_capt = 0

        for nome in todos:

            if nome not in corretores:
                corretores[nome] = {
                    "vgv": 0,
                    "vgc": 0,
                    "imoveis": {}
                }

            corretores[nome]["vgv"] += valor_imovel

        # VENDA
        if vendedores and parcela_venda:

            por_vendedor = parcela_venda / len(vendedores)

            for nome in vendedores:

                corretores[nome]["vgc"] += por_vendedor

                if id_contrato not in corretores[nome]["imoveis"]:
                    corretores[nome]["imoveis"][id_contrato] = {
                        "valor": valor_imovel,
                        "comissao": 0
                    }

                corretores[nome]["imoveis"][id_contrato]["comissao"] += por_vendedor

        # CAPTAÇÃO
        if captadores and parcela_capt:

            por_capt = parcela_capt / len(captadores)

            for nome in captadores:

                corretores[nome]["vgc"] += por_capt

                if id_contrato not in corretores[nome]["imoveis"]:
                    corretores[nome]["imoveis"][id_contrato] = {
                        "valor": valor_imovel,
                        "comissao": 0
                    }

                corretores[nome]["imoveis"][id_contrato]["comissao"] += por_capt

    # converter dict para lista
    for nome in corretores:

        corretores[nome]["imoveis"] = [
            {
                "contrato": contrato,
                "valor": dados["valor"],
                "comissao": dados["comissao"]
            }
            for contrato, dados in corretores[nome]["imoveis"].items()
        ]

    return corretores


# ==========================================================
# PDF POR CORRETOR
# ==========================================================

def gerar_pdf_corretores(corretores,caminho_pdf):

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    styles=getSampleStyleSheet()

    doc=SimpleDocTemplate(caminho_pdf,pagesize=A4)

    story=[]

    for nome,dados in sorted(corretores.items()):

        story.append(Paragraph(f"<b>{nome}</b>",styles["Title"]))

        story.append(Spacer(1,10))

        story.append(Paragraph(f"VGV Geral: R$ {dados['vgv']:,.2f}",styles["Normal"]))

        story.append(Paragraph(f"VGC Geral: R$ {dados['vgc']/0.06:,.2f}",styles["Normal"]))

        story.append(Spacer(1,10))

        tabela=[["Imóvel","Valor","Comissão"]]

        for im in dados["imoveis"]:

            tabela.append([
                im["contrato"],
                f"R$ {im['valor']:,.2f}",
                f"R$ {im['comissao']:,.2f}"
            ])

        t=Table(tabela,colWidths=[260,120,120])

        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.grey),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),0.5,colors.black)
        ]))

        story.append(t)

        story.append(PageBreak())

    doc.build(story)


# ==========================================================
# EXECUÇÃO
# ==========================================================

def main(file_path=None):

    if file_path:
        df=pd.read_excel(file_path)
    else:
        df=carregar_dados_sheets()

    cols_fin=['Valor_Negocio','Valor_Total_61']

    for col in cols_fin:
        df[col]=df[col].apply(_to_float_br)

    df['Data_Contrato']=pd.to_datetime(df['Data_Contrato'],errors='coerce')

    df_mes=df[
        (df['Data_Contrato']>='2026-01-01') &
        (df['Data_Contrato']<='2026-03-18')
    ].copy()

    res_geral=processar_rankings(df_mes,'CORRETOR')

    print("\nRANKING VGV GERAL\n")

    for n,v in sorted(res_geral["VGV_GERAL"].items(),key=lambda x:x[1],reverse=True):
        print(n,v)

    print("\nGerando relatório por corretor...")

    corretores=extrair_imoveis_por_corretor(df_mes)

    gerar_pdf_corretores(
        corretores,
        f"relatorio_corretores_{ANO_RELATORIO}_{MES_RELATORIO:02d}.pdf"
    )

    print("PDF gerado.")


if __name__ == "__main__":

    main()