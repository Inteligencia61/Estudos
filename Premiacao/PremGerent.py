# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
import calendar

# Tenta importar gspread, mas permite execução local sem ele
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

SHEET_ID_CONTRATOS = "1I9Lnbf3Be6oz9YPlHFiA9PkFFb9svDWvDtIcHH5I2QY"
SHEET_ID_BASE_INTELIGENCIA = "1_GA3LfjgQDTR_oly9fw5-XwHHTMWaUJixdVZ4PIHPB8"

ABA_CONTRATOS = "Vendas"
ABA_DIM_CORRETOR = "Dim_Corretor"
ABA_DIM_GERENTE = "Dim_Gerente"


# ==========================================================
# FUNÇÕES DE APOIO
# ==========================================================
def _to_float_br(x) -> float:
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
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    try:
        return float(s)
    except:
        return 0.0


def limpar_nome(n):
    if pd.isna(n) or str(n).strip() in ["", "-", "nan", "NAN", "None"]:
        return ""
    return str(n).strip().upper()


def formatar_moeda_br(valor):
    return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def valor_positivo(x):
    if pd.isna(x):
        return False
    try:
        return float(_to_float_br(x)) > 0
    except:
        return False


def autenticar_google_sheets():
    if not gspread:
        raise ImportError("Biblioteca gspread não instalada.")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("../cred.json", scope)
    client = gspread.authorize(creds)
    return client


def carregar_aba_por_id(sheet_id, nome_aba):
    client = autenticar_google_sheets()
    planilha = client.open_by_key(sheet_id)
    aba = planilha.worksheet(nome_aba)
    dados = aba.get_all_values()

    if not dados:
        return pd.DataFrame()

    header = dados[0]
    rows = dados[1:]

    df = pd.DataFrame(rows, columns=header)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    return df


# ==========================================================
# CARREGAMENTO DAS BASES
# ==========================================================
def carregar_contratos(sheet_id_contratos):
    df = carregar_aba_por_id(sheet_id_contratos, ABA_CONTRATOS)
    return df


def carregar_base_inteligencia(sheet_id_base):
    df_corretor = carregar_aba_por_id(sheet_id_base, ABA_DIM_CORRETOR)
    df_gerente = carregar_aba_por_id(sheet_id_base, ABA_DIM_GERENTE)
    return df_corretor, df_gerente


# ==========================================================
# MAPA CORRETOR -> GERENTE
# ==========================================================
def montar_mapa_corretor_gerente(df_dim_corretor, df_dim_gerente):
    """
    Retorna:
    - mapa_nome_corretor_para_gerente_nome
    - mapa_nome_corretor_para_id_gerente
    """
    dfc = df_dim_corretor.copy()
    dfg = df_dim_gerente.copy()

    dfc.columns = [str(c).strip() for c in dfc.columns]
    dfg.columns = [str(c).strip() for c in dfg.columns]

    # normalizações
    dfc["Nome_norm"] = dfc["Nome"].apply(limpar_nome)
    dfc["IdGerente_norm"] = dfc["IdGerente"].astype(str).str.strip().str.upper()

    dfg["IdGerente_norm"] = dfg["IdGerente"].astype(str).str.strip().str.upper()
    dfg["NomeGerente_norm"] = dfg["Nome"].apply(limpar_nome)

    mapa_idgerente_para_nome = dict(zip(dfg["IdGerente_norm"], dfg["NomeGerente_norm"]))

    mapa_corretor_para_idgerente = {}
    mapa_corretor_para_nomegerente = {}

    for _, row in dfc.iterrows():
        nome_corretor = row.get("Nome_norm", "")
        id_gerente = row.get("IdGerente_norm", "")

        if not nome_corretor:
            continue

        nome_gerente = mapa_idgerente_para_nome.get(id_gerente, "")
        mapa_corretor_para_idgerente[nome_corretor] = id_gerente
        mapa_corretor_para_nomegerente[nome_corretor] = nome_gerente

    return mapa_corretor_para_nomegerente, mapa_corretor_para_idgerente


# ==========================================================
# RANKING DOS CORRETORES
# ==========================================================
def processar_rankings_corretores(df_subset):
    vgv_cap, vgv_vend, vgv_geral = {}, {}, {}
    vgc_cap, vgc_vend, vgc_geral = {}, {}, {}

    for _, row in df_subset.iterrows():
        v_imovel = _to_float_br(row.get("Valor_Negocio", 0))
        v_comissao_total = _to_float_br(row.get("Valor_Total_61", 0))

        v_nomes = {
            limpar_nome(row.get("Corretor_Venda_1_Nome")),
            limpar_nome(row.get("Corretor_Venda_2_Nome")),
        }
        c_nomes = {
            limpar_nome(row.get("Corretor_Captador_1_Nome")),
            limpar_nome(row.get("Corretor_Captador_2_Nome")),
        }

        v_nomes = {n for n in v_nomes if n}
        c_nomes = {n for n in c_nomes if n}
        todos = v_nomes | c_nomes

        for n in v_nomes:
            vgv_vend[n] = vgv_vend.get(n, 0) + v_imovel

        for n in c_nomes:
            vgv_cap[n] = vgv_cap.get(n, 0) + v_imovel

        for n in todos:
            vgv_geral[n] = vgv_geral.get(n, 0) + v_imovel

        tem_venda = len(v_nomes) > 0
        tem_capt = len(c_nomes) > 0

        if tem_venda and tem_capt:
            parcela_venda = v_comissao_total * 0.5
            parcela_capt = v_comissao_total * 0.5
        elif tem_venda:
            parcela_venda = v_comissao_total
            parcela_capt = 0
        elif tem_capt:
            parcela_venda = 0
            parcela_capt = v_comissao_total
        else:
            parcela_venda = parcela_capt = 0

        if tem_venda and parcela_venda:
            por_vendedor = parcela_venda / len(v_nomes)
            for n in v_nomes:
                vgc_vend[n] = vgc_vend.get(n, 0) + por_vendedor
                vgc_geral[n] = vgc_geral.get(n, 0) + por_vendedor

        if tem_capt and parcela_capt:
            por_capt = parcela_capt / len(c_nomes)
            for n in c_nomes:
                vgc_cap[n] = vgc_cap.get(n, 0) + por_capt
                vgc_geral[n] = vgc_geral.get(n, 0) + por_capt

    return {
        "VGV_CAP": vgv_cap,
        "VGV_VEND": vgv_vend,
        "VGV_GERAL": vgv_geral,
        "VGC_CAP": vgc_cap,
        "VGC_VEND": vgc_vend,
        "VGC_GERAL": vgc_geral,
    }


# ==========================================================
# NOVA LÓGICA DE GERENTES VIA BASE INTELIGÊNCIA
# ==========================================================
def processar_gerentes_via_dim_corretor(df_subset, mapa_corretor_gerente):
    """
    O gerente é definido pela Dim_Corretor/Dim_Gerente, não pelo nome do gerente da linha.
    O VGV do gerente é composto pelos corretores oficialmente vinculados a ele.
    O detalhamento é rateado para fechar com o VGV do gerente.
    """
    vgv_cap, vgv_vend, vgv_geral = {}, {}, {}
    vgc_cap, vgc_vend, vgc_geral = {}, {}, {}

    detalhes_vgv_cap = {}
    detalhes_vgv_vend = {}
    detalhes_vgv_geral = {}

    for _, row in df_subset.iterrows():
        v_imovel = _to_float_br(row.get("Valor_Negocio", 0))
        v_comissao_total = _to_float_br(row.get("Valor_Total_61", 0))

        # --- corretores válidos da linha por lado
        corretores_venda = []
        if valor_positivo(row.get("$_Corretor_Venda_1")):
            nome = limpar_nome(row.get("Corretor_Venda_1_Nome"))
            if nome:
                corretores_venda.append(nome)

        if valor_positivo(row.get("$_Corretor_Venda_2")):
            nome = limpar_nome(row.get("Corretor_Venda_2_Nome"))
            if nome and nome not in corretores_venda:
                corretores_venda.append(nome)

        corretores_capt = []
        if valor_positivo(row.get("$_Corretor_Captador_1")):
            nome = limpar_nome(row.get("Corretor_Captador_1_Nome"))
            if nome:
                corretores_capt.append(nome)

        if valor_positivo(row.get("$_Corretor_Captador_2")):
            nome = limpar_nome(row.get("Corretor_Captador_2_Nome"))
            if nome and nome not in corretores_capt:
                corretores_capt.append(nome)

        # --- gerente(s) da linha definidos pela base mestre
        gerentes_venda_dict = {}
        for corretor in corretores_venda:
            gerente = mapa_corretor_gerente.get(corretor, "")
            if gerente:
                gerentes_venda_dict.setdefault(gerente, []).append(corretor)

        gerentes_capt_dict = {}
        for corretor in corretores_capt:
            gerente = mapa_corretor_gerente.get(corretor, "")
            if gerente:
                gerentes_capt_dict.setdefault(gerente, []).append(corretor)

        # -------------------------
        # VGV VENDA
        # -------------------------
        for gerente, lista_corretores in gerentes_venda_dict.items():
            vgv_vend[gerente] = vgv_vend.get(gerente, 0) + v_imovel

            detalhes_vgv_vend.setdefault(gerente, {})
            quota = v_imovel / len(lista_corretores)

            for corretor in lista_corretores:
                detalhes_vgv_vend[gerente][corretor] = (
                    detalhes_vgv_vend[gerente].get(corretor, 0) + quota
                )

        # -------------------------
        # VGV CAPTAÇÃO
        # -------------------------
        for gerente, lista_corretores in gerentes_capt_dict.items():
            vgv_cap[gerente] = vgv_cap.get(gerente, 0) + v_imovel

            detalhes_vgv_cap.setdefault(gerente, {})
            quota = v_imovel / len(lista_corretores)

            for corretor in lista_corretores:
                detalhes_vgv_cap[gerente][corretor] = (
                    detalhes_vgv_cap[gerente].get(corretor, 0) + quota
                )

        # -------------------------
        # VGV GERAL
        # conta a linha 1 vez por gerente
        # e rateia dentro do conjunto de corretores daquele gerente na linha
        # -------------------------
        gerentes_geral_dict = {}

        for gerente, lista_corretores in gerentes_venda_dict.items():
            gerentes_geral_dict.setdefault(gerente, set()).update(lista_corretores)

        for gerente, lista_corretores in gerentes_capt_dict.items():
            gerentes_geral_dict.setdefault(gerente, set()).update(lista_corretores)

        for gerente, conjunto_corretores in gerentes_geral_dict.items():
            vgv_geral[gerente] = vgv_geral.get(gerente, 0) + v_imovel

            detalhes_vgv_geral.setdefault(gerente, {})
            lista = list(conjunto_corretores)
            quota = v_imovel / len(lista)

            for corretor in lista:
                detalhes_vgv_geral[gerente][corretor] = (
                    detalhes_vgv_geral[gerente].get(corretor, 0) + quota
                )

        # -------------------------
        # VGC VENDA
        # -------------------------
        if gerentes_venda_dict and gerentes_capt_dict:
            parcela_venda = v_comissao_total * 0.5
            parcela_capt = v_comissao_total * 0.5
        elif gerentes_venda_dict:
            parcela_venda = v_comissao_total
            parcela_capt = 0
        elif gerentes_capt_dict:
            parcela_venda = 0
            parcela_capt = v_comissao_total
        else:
            parcela_venda = parcela_capt = 0

        if gerentes_venda_dict and parcela_venda:
            quota_gerente_venda = parcela_venda / len(gerentes_venda_dict)
            for gerente in gerentes_venda_dict:
                vgc_vend[gerente] = vgc_vend.get(gerente, 0) + quota_gerente_venda
                vgc_geral[gerente] = vgc_geral.get(gerente, 0) + quota_gerente_venda

        if gerentes_capt_dict and parcela_capt:
            quota_gerente_capt = parcela_capt / len(gerentes_capt_dict)
            for gerente in gerentes_capt_dict:
                vgc_cap[gerente] = vgc_cap.get(gerente, 0) + quota_gerente_capt
                vgc_geral[gerente] = vgc_geral.get(gerente, 0) + quota_gerente_capt

    return {
        "VGV_CAP": vgv_cap,
        "VGV_VEND": vgv_vend,
        "VGV_GERAL": vgv_geral,
        "VGC_CAP": vgc_cap,
        "VGC_VEND": vgc_vend,
        "VGC_GERAL": vgc_geral,
        "DETALHES_VGV_CAP": detalhes_vgv_cap,
        "DETALHES_VGV_VEND": detalhes_vgv_vend,
        "DETALHES_VGV_GERAL": detalhes_vgv_geral,
    }


# ==========================================================
# PDF DE GERENTES
# ==========================================================
def gerar_pdf_gerentes_detalhado(res_gerentes: dict, caminho_pdf: str, titulo: str, top_n: int = 30):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    doc = SimpleDocTemplate(
        caminho_pdf,
        pagesize=A4,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(titulo, styles["Title"]))
    story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    ranking_gerentes = sorted(
        res_gerentes["VGV_GERAL"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    story.append(Paragraph("Ranking de Gerentes - VGV Geral", styles["Heading2"]))

    if not ranking_gerentes:
        story.append(Paragraph("Sem dados.", styles["Normal"]))
        doc.build(story)
        return

    resumo = [["Pos", "Gerente", "VGV Geral"]]
    for i, (gerente, valor) in enumerate(ranking_gerentes, start=1):
        resumo.append([str(i), gerente, formatar_moeda_br(valor)])

    t = Table(resumo, hAlign="LEFT", colWidths=[35, 290, 120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    detalhes = res_gerentes.get("DETALHES_VGV_GERAL", {})

    for i, (gerente, valor_total) in enumerate(ranking_gerentes, start=1):
        story.append(Paragraph(f"{i}. {gerente} - {formatar_moeda_br(valor_total)}", styles["Heading2"]))
        story.append(Spacer(1, 6))

        corretores = detalhes.get(gerente, {})
        corretores_ordenados = sorted(corretores.items(), key=lambda x: x[1], reverse=True)

        data = [["Pos", "Corretor", "VGV usado na soma"]]
        soma = 0.0

        for j, (corretor, valor) in enumerate(corretores_ordenados, start=1):
            soma += float(valor or 0)
            data.append([str(j), corretor, formatar_moeda_br(valor)])

        data.append(["", "TOTAL DETALHADO", formatar_moeda_br(soma)])

        td = Table(data, hAlign="LEFT", colWidths=[35, 290, 120])
        td.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.beige, colors.whitesmoke]),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#d1d5db")),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ]))
        story.append(td)
        story.append(Spacer(1, 14))

        if i < len(ranking_gerentes):
            story.append(PageBreak())

    doc.build(story)


# ==========================================================
# MAIN
# ==========================================================
def main(sheet_id_contratos=SHEET_ID_CONTRATOS, sheet_id_base=SHEET_ID_BASE_INTELIGENCIA):
    # 1) Carrega dados
    df = carregar_contratos(sheet_id_contratos)
    df_dim_corretor, df_dim_gerente = carregar_base_inteligencia(sheet_id_base)

    # 2) Converte financeiros
    for col in ["Valor_Negocio", "Valor_Total_61"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_float_br)

    # 3) Filtro do mês
    df["Data_Contrato"] = pd.to_datetime(df["Data_Contrato"], errors="coerce")
    ultimo_dia = calendar.monthrange(ANO_RELATORIO, MES_RELATORIO)[1]

    df_mes = df[
        (df["Data_Contrato"] >= f"{ANO_RELATORIO}-{MES_RELATORIO:02d}-01") &
        (df["Data_Contrato"] <= f"{ANO_RELATORIO}-{MES_RELATORIO:02d}-{ultimo_dia}")
    ].copy()

    # 4) Monta mapa oficial corretor -> gerente
    mapa_corretor_gerente, _ = montar_mapa_corretor_gerente(df_dim_corretor, df_dim_gerente)

    # 5) Processa corretores
    res_corretores = processar_rankings_corretores(df_mes)

    # 6) Processa gerentes via base mestre
    res_gerentes = processar_gerentes_via_dim_corretor(df_mes, mapa_corretor_gerente)

    # 7) Saída console
    print("\n>>> GERENTES - VGV GERAL")
    for nome, valor in sorted(res_gerentes["VGV_GERAL"].items(), key=lambda x: x[1], reverse=True):
        print(f"{nome.ljust(30)} | {formatar_moeda_br(valor)}")

    # 8) PDF
    nome_pdf_gerentes = f"ranking_gerentes_detalhado_{ANO_RELATORIO}_{MES_RELATORIO:02d}.pdf"
    gerar_pdf_gerentes_detalhado(
        res_gerentes,
        nome_pdf_gerentes,
        f"Ranking de Gerentes com base na Dim_Corretor - {MES_RELATORIO:02d}/{ANO_RELATORIO}",
        top_n=TOP_N
    )

    print("\nArquivo gerado:")
    print(f"- {nome_pdf_gerentes}")


if __name__ == "__main__":
    main()