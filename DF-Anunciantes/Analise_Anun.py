import os
import re
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT


# =========================
# CONFIG
# =========================
CSV_DF = "./2026-03-02 DF.csv"
CSV_WI = "./2026-03-02 WI real.csv"

NOME_DF = "DF"
NOME_WI = "WI"

OUTPUT_PDF = "./comparacao_creci_df_wi.pdf"
OUTPUT_DIR = "./saida_comparacao_creci"

MIN_DIGITOS_CRECI = 4


# =========================
# FUNÇÕES AUXILIARES
# =========================
def normalizar_texto(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = (
        s.replace("Á", "A").replace("À", "A").replace("Ã", "A").replace("Â", "A")
         .replace("É", "E").replace("Ê", "E")
         .replace("Í", "I")
         .replace("Ó", "O").replace("Õ", "O").replace("Ô", "O")
         .replace("Ú", "U")
         .replace("Ç", "C")
    )
    s = re.sub(r"\s+", " ", s)
    return s


def normalizar_creci(x):
    """
    Mantém apenas dígitos.
    Ex:
    '25.432' -> '25432'
    'CRECI 1234' -> '1234'
    """
    if pd.isna(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    s = s.replace("'", "").replace('"', "")
    s = re.sub(r"[^\d]", "", s)

    if not s:
        return None

    return s


def creci_valido(creci):
    return creci is not None and len(str(creci)) >= MIN_DIGITOS_CRECI


def detectar_separador(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        amostra = f.read(5000)
    return ";" if amostra.count(";") > amostra.count(",") else ","


def ler_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    sep = detectar_separador(path)

    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    obrigatorias = {"codigo", "creci", "anunciante"}
    faltantes = obrigatorias - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes em {path}: {faltantes}")

    df["codigo_norm"] = df["codigo"].astype(str).str.strip()
    df["anunciante_norm"] = df["anunciante"].apply(normalizar_texto)
    df["creci_norm"] = df["creci"].apply(normalizar_creci)
    df["creci_valido"] = df["creci_norm"].apply(creci_valido)

    return df


def consolidar_por_creci(df, origem):
    """
    Consolida por CRECI válido.
    """
    base = df[df["creci_valido"]].copy()

    agrupado = (
        base.groupby("creci_norm", dropna=False)
        .agg(
            anunciante_principal=("anunciante", lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna()) else ""),
            qtd_imoveis=("codigo_norm", "nunique")
        )
        .reset_index()
    )

    agrupado["origem"] = origem
    return agrupado


def resumir_invalidos(df, origem):
    invalidos = df[~df["creci_valido"]].copy()

    resumo = (
        invalidos.groupby(["anunciante"], dropna=False)
        .agg(
            qtd_imoveis=("codigo_norm", "nunique"),
            qtd_registros=("codigo_norm", "count")
        )
        .reset_index()
        .sort_values(["qtd_imoveis", "qtd_registros"], ascending=False)
    )
    resumo["origem"] = origem

    return invalidos, resumo


def dataframe_para_tabela(df, max_rows=40):
    """
    Converte DataFrame em lista para tabela do ReportLab.
    Limita linhas para o PDF não explodir.
    """
    if df.empty:
        return [["Sem registros"]]

    df_show = df.head(max_rows).copy()
    data = [list(df_show.columns)] + df_show.astype(str).values.tolist()

    if len(df) > max_rows:
        data.append([f"... total de {len(df)} registros. Exibindo apenas os primeiros {max_rows}."] + [""] * (len(df_show.columns) - 1))

    return data


def criar_tabela_pdf(data, col_widths=None):
    tabela = Table(data, colWidths=col_widths, repeatRows=1)

    estilo = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])
    tabela.setStyle(estilo)
    return tabela


# =========================
# GERAÇÃO PDF
# =========================
def gerar_pdf(relatorio_path, so_df, so_wi, resumo_invalidos_df, resumo_invalidos_wi, totais):
    styles = getSampleStyleSheet()

    titulo_style = styles["Title"]
    subtitulo_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    normal_style.fontName = "Helvetica"
    normal_style.fontSize = 10
    normal_style.leading = 14

    destaque_style = ParagraphStyle(
        "Destaque",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
    )

    doc = SimpleDocTemplate(
        relatorio_path,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    story = []

    story.append(Paragraph("Relatório Comparativo de CRECI - DF x WI", titulo_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Resumo Geral", subtitulo_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"Total de registros em DF: <b>{totais['df_total']}</b>", normal_style))
    story.append(Paragraph(f"Total de registros em WI: <b>{totais['wi_total']}</b>", normal_style))
    story.append(Paragraph(f"CRECI válidos em DF: <b>{totais['df_validos']}</b>", normal_style))
    story.append(Paragraph(f"CRECI válidos em WI: <b>{totais['wi_validos']}</b>", normal_style))
    story.append(Paragraph(f"CRECI inválidos em DF: <b>{totais['df_invalidos']}</b>", normal_style))
    story.append(Paragraph(f"CRECI inválidos em WI: <b>{totais['wi_invalidos']}</b>", normal_style))
    story.append(Paragraph(f"CRECI somente em DF: <b>{totais['so_df']}</b>", normal_style))
    story.append(Paragraph(f"CRECI somente em WI: <b>{totais['so_wi']}</b>", normal_style))
    story.append(Spacer(1, 18))

    story.append(Paragraph(
        "Regra de validade adotada: o CRECI foi considerado válido apenas quando possui no mínimo 4 dígitos numéricos após a normalização.",
        destaque_style
    ))
    story.append(PageBreak())

    # CRECI só no DF
    story.append(Paragraph("CRECI presentes em DF e ausentes em WI", subtitulo_style))
    story.append(Spacer(1, 8))
    data_df = dataframe_para_tabela(
        so_df[["creci_norm", "anunciante_principal_DF", "qtd_imoveis_DF"]]
        if not so_df.empty else pd.DataFrame()
    )
    story.append(criar_tabela_pdf(data_df, col_widths=[100, 260, 100]))
    story.append(Spacer(1, 16))

    # CRECI só no WI
    story.append(Paragraph("CRECI presentes em WI e ausentes em DF", subtitulo_style))
    story.append(Spacer(1, 8))
    data_wi = dataframe_para_tabela(
        so_wi[["creci_norm", "anunciante_principal_WI", "qtd_imoveis_WI"]]
        if not so_wi.empty else pd.DataFrame()
    )
    story.append(criar_tabela_pdf(data_wi, col_widths=[100, 260, 100]))
    story.append(PageBreak())

    # Inválidos DF
    story.append(Paragraph("Anunciantes com CRECI inválido - DF", subtitulo_style))
    story.append(Spacer(1, 8))
    data_inv_df = dataframe_para_tabela(
        resumo_invalidos_df[["anunciante", "qtd_imoveis", "qtd_registros"]]
        if not resumo_invalidos_df.empty else pd.DataFrame()
    )
    story.append(criar_tabela_pdf(data_inv_df, col_widths=[300, 100, 100]))
    story.append(Spacer(1, 16))

    # Inválidos WI
    story.append(Paragraph("Anunciantes com CRECI inválido - WI", subtitulo_style))
    story.append(Spacer(1, 8))
    data_inv_wi = dataframe_para_tabela(
        resumo_invalidos_wi[["anunciante", "qtd_imoveis", "qtd_registros"]]
        if not resumo_invalidos_wi.empty else pd.DataFrame()
    )
    story.append(criar_tabela_pdf(data_inv_wi, col_widths=[300, 100, 100]))

    doc.build(story)


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_df = ler_csv(CSV_DF)
    df_wi = ler_csv(CSV_WI)

    # inválidos
    invalidos_df, resumo_invalidos_df = resumir_invalidos(df_df, NOME_DF)
    invalidos_wi, resumo_invalidos_wi = resumir_invalidos(df_wi, NOME_WI)

    # válidos consolidados
    cons_df = consolidar_por_creci(df_df, NOME_DF)
    cons_wi = consolidar_por_creci(df_wi, NOME_WI)

    # comparação por CRECI
    comp = cons_df.merge(
        cons_wi,
        on="creci_norm",
        how="outer",
        suffixes=("_DF", "_WI"),
        indicator=True
    )

    so_df = comp[comp["_merge"] == "left_only"].copy()
    so_wi = comp[comp["_merge"] == "right_only"].copy()
    ambos = comp[comp["_merge"] == "both"].copy()

    # ordenar
    if not so_df.empty:
        so_df = so_df.sort_values(["qtd_imoveis_DF", "creci_norm"], ascending=[False, True])

    if not so_wi.empty:
        so_wi = so_wi.sort_values(["qtd_imoveis_WI", "creci_norm"], ascending=[False, True])

    totais = {
        "df_total": len(df_df),
        "wi_total": len(df_wi),
        "df_validos": int(df_df["creci_valido"].sum()),
        "wi_validos": int(df_wi["creci_valido"].sum()),
        "df_invalidos": int((~df_df["creci_valido"]).sum()),
        "wi_invalidos": int((~df_wi["creci_valido"]).sum()),
        "so_df": len(so_df),
        "so_wi": len(so_wi),
        "ambos": len(ambos),
    }

    # salvar CSVs auxiliares
    so_df.to_csv(os.path.join(OUTPUT_DIR, "creci_so_em_df.csv"), index=False, encoding="utf-8-sig")
    so_wi.to_csv(os.path.join(OUTPUT_DIR, "creci_so_em_wi.csv"), index=False, encoding="utf-8-sig")
    ambos.to_csv(os.path.join(OUTPUT_DIR, "creci_em_ambos.csv"), index=False, encoding="utf-8-sig")
    invalidos_df.to_csv(os.path.join(OUTPUT_DIR, "creci_invalidos_df.csv"), index=False, encoding="utf-8-sig")
    invalidos_wi.to_csv(os.path.join(OUTPUT_DIR, "creci_invalidos_wi.csv"), index=False, encoding="utf-8-sig")
    resumo_invalidos_df.to_csv(os.path.join(OUTPUT_DIR, "resumo_invalidos_df.csv"), index=False, encoding="utf-8-sig")
    resumo_invalidos_wi.to_csv(os.path.join(OUTPUT_DIR, "resumo_invalidos_wi.csv"), index=False, encoding="utf-8-sig")

    # gerar PDF
    gerar_pdf(
        OUTPUT_PDF,
        so_df,
        so_wi,
        resumo_invalidos_df,
        resumo_invalidos_wi,
        totais
    )

    print("=" * 80)
    print("PROCESSO FINALIZADO")
    print("=" * 80)
    print(f"PDF gerado: {OUTPUT_PDF}")
    print(f"Arquivos auxiliares gerados em: {OUTPUT_DIR}")
    print()
    print(f"CRECI só no DF: {len(so_df)}")
    print(f"CRECI só no WI: {len(so_wi)}")
    print(f"CRECI inválidos no DF: {len(invalidos_df)}")
    print(f"CRECI inválidos no WI: {len(invalidos_wi)}")


if __name__ == "__main__":
    main()