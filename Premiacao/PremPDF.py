# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from datetime import datetime

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
TOP_N = 30
GERENTES_ESPECIAIS = ['HELIO', 'LUANA']
EXCECAO_OUTROS = ['FERNANDA LINDSAY']

# ==========================================================
# FUNÇÕES DE APOIO (BASEADAS NO SEU EXEMPLO)
# ==========================================================
def _to_float_br(x) -> float:
    """
    Função de conversão idêntica ao seu exemplo que funcionou.
    """
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

        # --- RANKINGS VGV ---
        for n in v_nomes:
            vgv_vend[n] = vgv_vend.get(n, 0) + v_imovel
        for n in c_nomes:
            vgv_cap[n] = vgv_cap.get(n, 0) + v_imovel
        for n in todos_nomes:
            vgv_geral[n] = vgv_geral.get(n, 0) + v_imovel

        # --- RANKINGS VGC (divide por lado e por pessoa) ---
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
        for n in v_nomes:
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
# NOVO: COMBINAÇÃO (AC + PP) PARA GERAR O GERAL
# ==========================================================
def somar_rankings(*dicts: dict) -> dict:
    """
    Soma vários dicts {nome: valor}.
    """
    out = {}
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + float(v or 0.0)
    return out

def combinar_resultados(res_a: dict, res_b: dict) -> dict:
    """
    Combina dois resultados no formato retornado por processar_rankings.
    """
    keys = ['VGV_CAP','VGV_VEND','VGV_GERAL','VGC_CAP','VGC_VEND','VGC_GERAL']
    return {k: somar_rankings(res_a.get(k, {}), res_b.get(k, {})) for k in keys}

# ==========================================================
# TABELAS PARA EXPORTAÇÃO
# ==========================================================
def ranking_dict_to_df(d: dict, top_n: int = 30) -> pd.DataFrame:
    if not d:
        return pd.DataFrame(columns=["Pos", "Nome", "Valor"])
    itens = sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]
    df = pd.DataFrame(itens, columns=["Nome", "Valor"])
    df.insert(0, "Pos", range(1, len(df) + 1))
    return df

def montar_tabelas_relatorio(res_esp, res_outros, res_total, res_gerentes, top_n=30):
    """
    Retorna um dict {nome_da_secao: dataframe} pronto para exportar.
    """
    tabelas = {}

    # CORRETORES - AC (Hélio/Luana)
    tabelas["CORRETORES_AC_VGV_CAP"]   = ranking_dict_to_df(res_esp["VGV_CAP"], top_n)
    tabelas["CORRETORES_AC_VGV_VEND"]  = ranking_dict_to_df(res_esp["VGV_VEND"], top_n)
    tabelas["CORRETORES_AC_VGV_GERAL"] = ranking_dict_to_df(res_esp["VGV_GERAL"], top_n)

    tabelas["CORRETORES_AC_VGC_CAP"]   = ranking_dict_to_df(res_esp["VGC_CAP"], top_n)
    tabelas["CORRETORES_AC_VGC_VEND"]  = ranking_dict_to_df(res_esp["VGC_VEND"], top_n)
    tabelas["CORRETORES_AC_VGC_GERAL"] = ranking_dict_to_df(res_esp["VGC_GERAL"], top_n)

    # CORRETORES - PP (Outros)
    tabelas["CORRETORES_PP_VGV_CAP"]   = ranking_dict_to_df(res_outros["VGV_CAP"], top_n)
    tabelas["CORRETORES_PP_VGV_VEND"]  = ranking_dict_to_df(res_outros["VGV_VEND"], top_n)
    tabelas["CORRETORES_PP_VGV_GERAL"] = ranking_dict_to_df(res_outros["VGV_GERAL"], top_n)

    tabelas["CORRETORES_PP_VGC_CAP"]   = ranking_dict_to_df(res_outros["VGC_CAP"], top_n)
    tabelas["CORRETORES_PP_VGC_VEND"]  = ranking_dict_to_df(res_outros["VGC_VEND"], top_n)
    tabelas["CORRETORES_PP_VGC_GERAL"] = ranking_dict_to_df(res_outros["VGC_GERAL"], top_n)

    # NOVO: CORRETORES - GERAL (AC + PP)
    tabelas["CORRETORES_GERAL_VGV_CAP"]   = ranking_dict_to_df(res_total["VGV_CAP"], top_n)
    tabelas["CORRETORES_GERAL_VGV_VEND"]  = ranking_dict_to_df(res_total["VGV_VEND"], top_n)
    tabelas["CORRETORES_GERAL_VGV_GERAL"] = ranking_dict_to_df(res_total["VGV_GERAL"], top_n)

    tabelas["CORRETORES_GERAL_VGC_CAP"]   = ranking_dict_to_df(res_total["VGC_CAP"], top_n)
    tabelas["CORRETORES_GERAL_VGC_VEND"]  = ranking_dict_to_df(res_total["VGC_VEND"], top_n)
    tabelas["CORRETORES_GERAL_VGC_GERAL"] = ranking_dict_to_df(res_total["VGC_GERAL"], top_n)

    # GERENTES
    tabelas["GERENTES_VGV_GERAL"] = ranking_dict_to_df(res_gerentes["VGV_GERAL"], top_n)
    tabelas["GERENTES_VGC_GERAL"] = ranking_dict_to_df(res_gerentes["VGC_GERAL"], top_n)
    tabelas["GERENTES_VGC_CAP"]   = ranking_dict_to_df(res_gerentes["VGC_CAP"], top_n)
    tabelas["GERENTES_VGC_VEND"]  = ranking_dict_to_df(res_gerentes["VGC_VEND"], top_n)

    return tabelas

# ==========================================================
# EXPORTAÇÃO PDF / DOCX
# ==========================================================
def gerar_pdf(tabelas: dict, caminho_pdf: str, titulo: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    doc = SimpleDocTemplate(
        caminho_pdf,
        pagesize=A4,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(titulo, styles["Title"]))
    story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    def add_table(df: pd.DataFrame, secao: str):
        story.append(Paragraph(secao.replace("_", " "), styles["Heading2"]))
        if df.empty:
            story.append(Paragraph("Sem dados.", styles["Normal"]))
            story.append(Spacer(1, 10))
            return

        df2 = df.copy()
        df2["Valor"] = df2["Valor"].apply(lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        data = [df2.columns.tolist()] + df2.values.tolist()

        t = Table(data, hAlign="LEFT", colWidths=[35, 260, 120])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
            ("FONTSIZE", (0,1), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

    for i, (secao, df) in enumerate(tabelas.items()):
        add_table(df, secao)
        if (i + 1) % 3 == 0 and (i + 1) < len(tabelas):
            story.append(PageBreak())

    doc.build(story)

def gerar_docx(tabelas: dict, caminho_docx: str, titulo: str):
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    h = doc.add_heading(titulo, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    p.runs[0].font.size = Pt(10)

    doc.add_paragraph("")

    for secao, df in tabelas.items():
        doc.add_heading(secao.replace("_", " "), level=2)

        if df.empty:
            doc.add_paragraph("Sem dados.")
            doc.add_paragraph("")
            continue

        table = doc.add_table(rows=1, cols=3)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Pos"
        hdr_cells[1].text = "Nome"
        hdr_cells[2].text = "Valor"

        for _, r in df.iterrows():
            row_cells = table.add_row().cells
            row_cells[0].text = str(int(r["Pos"]))
            row_cells[1].text = str(r["Nome"])
            row_cells[2].text = f"R$ {float(r['Valor']):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

        doc.add_paragraph("")

    doc.save(caminho_docx)

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

    # 1) Limpeza Financeira
    cols_fin = ['Valor_Negocio', 'Valor_Total_61']
    for col in cols_fin:
        if col in df.columns:
            df[col] = df[col].apply(_to_float_br)

    # 2) Filtro de Data
    df['Data_Contrato'] = pd.to_datetime(df['Data_Contrato'], errors='coerce')
    df_mes = df[(df['Data_Contrato'] >= '2026-01-01') & (df['Data_Contrato'] <= '2026-01-31')].copy()

    # 3) Separação de Equipes (AC vs PP)
    def is_especial(row):
        gv = limpar_nome(row.get('Gerente_Venda_Nome'))
        gc = limpar_nome(row.get('Gerente_Captacao_Nome'))
        return any(g in gv or g in gc for g in GERENTES_ESPECIAIS)

    mask_especial = df_mes.apply(is_especial, axis=1)
    df_esp_raw = df_mes[mask_especial].copy()
    df_outros_raw = df_mes[~mask_especial].copy()

    # 4) Processamento dos Grupos
    res_esp = processar_rankings(df_esp_raw, 'CORRETOR')
    res_outros = processar_rankings(df_outros_raw, 'CORRETOR')
    res_gerentes = processar_rankings(df_mes, 'GERENTE')

    # 5) Ajuste Fernanda Lindsay (sempre vai para "Outros"/PP)
    for rank_tipo in ['VGV_CAP', 'VGV_VEND', 'VGV_GERAL', 'VGC_CAP', 'VGC_VEND', 'VGC_GERAL']:
        for nome in EXCECAO_OUTROS:
            if nome in res_esp[rank_tipo]:
                valor = res_esp[rank_tipo].pop(nome)
                res_outros[rank_tipo][nome] = res_outros[rank_tipo].get(nome, 0) + valor

    # 6) NOVO: Ranking GERAL (AC + PP)
    res_total = combinar_resultados(res_esp, res_outros)

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

    # NOVO: GERAL AC+PP (VGV)
    print("\n>>> GERAL AC + PP (VGV)")
    imprimir_ranking("VGV CAPTAÇÃO - GERAL", res_total['VGV_CAP'])
    imprimir_ranking("VGV VENDA - GERAL", res_total['VGV_VEND'])
    imprimir_ranking("VGV GERAL - GERAL", res_total['VGV_GERAL'])

    print("\n>>> EQUIPE HELIO / LUANA (VGC)")
    imprimir_ranking("VGC CAPTAÇÃO - AC", res_esp['VGC_CAP'])
    imprimir_ranking("VGC VENDA - AC", res_esp['VGC_VEND'])
    imprimir_ranking("VGC GERAL - AC", res_esp['VGC_GERAL'])

    print("\n>>> EQUIPE OUTROS (VGC)")
    imprimir_ranking("VGC CAPTAÇÃO - PP", res_outros['VGC_CAP'])
    imprimir_ranking("VGC VENDA - PP", res_outros['VGC_VEND'])
    imprimir_ranking("VGC GERAL - PP", res_outros['VGC_GERAL'])

    # NOVO: GERAL AC+PP (VGC)
    print("\n>>> GERAL AC + PP (VGC)")
    imprimir_ranking("VGC CAPTAÇÃO - GERAL", res_total['VGC_CAP'])
    imprimir_ranking("VGC VENDA - GERAL", res_total['VGC_VEND'])
    imprimir_ranking("VGC GERAL - GERAL", res_total['VGC_GERAL'])

    print("\n>>> RANKINGS GERENTES")
    imprimir_ranking("VGV GERAL - GERENTES", res_gerentes['VGV_GERAL'])
    imprimir_ranking("VGC GERAL - GERENTES", res_gerentes['VGC_GERAL'])
    imprimir_ranking("VGC CAPTAÇÃO - GERENTES", res_gerentes['VGC_CAP'])
    imprimir_ranking("VGC VENDA - GERENTES", res_gerentes['VGC_VEND'])

    # ==========================================================
    # GERAR ARQUIVOS (PDF e WORD)
    # ==========================================================
    titulo = f"Relatório de Rankings - {MES_RELATORIO:02d}/{ANO_RELATORIO}"

    tabelas = montar_tabelas_relatorio(res_esp, res_outros, res_total, res_gerentes, top_n=TOP_N)

    gerar_pdf(tabelas, f"ranking_{ANO_RELATORIO}_{MES_RELATORIO:02d}.pdf", titulo)
    gerar_docx(tabelas, f"ranking_{ANO_RELATORIO}_{MES_RELATORIO:02d}.docx", titulo)

    print("\nArquivos gerados:")
    print(f"- ranking_{ANO_RELATORIO}_{MES_RELATORIO:02d}.pdf")
    print(f"- ranking_{ANO_RELATORIO}_{MES_RELATORIO:02d}.docx")

if __name__ == "__main__":
    # main("/caminho/para/arquivo.xlsx")
    main()