# -*- coding: utf-8 -*-
"""
Ingestão de coletas — camada única de entrada da tabela `imoveis`.

Substitui a lista `CARGAS` editada à mão em `enviar_BD.py`. Descobre os
arquivos, deduz portal e data, normaliza para o schema canônico e carrega de
forma idempotente.

ESTRUTURA DE PASTAS (nada de nome de arquivo dentro de código):

    dados/coletas/<portal>/AAAA-MM-DD.csv

        dados/coletas/df/2026-09-01.csv   -> portal "df"
        dados/coletas/wi/2026-09-01.csv   -> portal "wi"

A raiz vem de DATA_DIR no .env (padrão: ./dados).

-------------------------------------------------------------------------
O CONTRATO DO DATAFRAME
-------------------------------------------------------------------------
Os scrapers mudaram de layout e o banco pagou por isso: hoje `oferta` guarda
"APARTAMENTO" em 31 mil linhas e `tipo` guarda "VENDA" em 178 mil, porque o
CSV novo chama de `tipo` o que o banco chama de `oferta`.

    layout ANTIGO  (alimentou o banco até aqui)
        codigo, link, creci, anunciante, oferta, tipo, area_util, ...
        oferta = VENDA/ALUGUEL      tipo = APARTAMENTO/CASA/...

    layout NOVO    (scraperDF.py e scraperWI.py de hoje)
        id, link, codigo, creci, anunciante, tipo, tipo_imovel, ..., data
        tipo   = venda/aluguel/lancamento     <- isto é OFERTA
        tipo_imovel = apartamento/casa/...    <- isto é TIPO

Este módulo detecta os dois e converte para o canônico abaixo. Nenhum outro
script precisa saber que existe mais de um layout.

    portal       text   obrigatório   df | wi
    id_anuncio   text   obrigatório   id do anúncio no portal (chave estável)
    codigo       text                 código da imobiliária; cai para id_anuncio
    oferta       text   obrigatório   VENDA | ALUGUEL
    tipo         text   obrigatório   APARTAMENTO | CASA | CASA CONDOMINIO | ...
    data_coleta  date   obrigatório   da coluna `data` ou do nome do arquivo
    preco, area_util, valor_m2, quartos, vagas, latitude, longitude  numéricos
    bairro, cidade, quadra, creci, anunciante, link                  texto

Uso:
    python BD/ingestao.py --dry-run        # o que entraria, sem gravar
    python BD/ingestao.py                  # carrega o pendente
    python BD/ingestao.py --arquivo dados/coletas/wi/2026-09-01.csv --portal wi
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql as psql
from psycopg2.extras import execute_values

load_dotenv()

TABELA = "imoveis"
SCHEMA_CTRL = "analytics"
TBL_CARGAS = "cargas"

PORTAIS_CONHECIDOS = {"df", "wi"}
RE_DATA = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Vocabulário de oferta. "LANCAMENTO" é venda: imóvel novo continua sendo venda,
# e separá-lo criava um terceiro balde que nenhum estudo consumia.
MAPA_OFERTA = {
    "VENDA": "VENDA", "VENDAS": "VENDA", "LANCAMENTO": "VENDA",
    "LANÇAMENTO": "VENDA", "COMPRA": "VENDA",
    "ALUGUEL": "ALUGUEL", "LOCACAO": "ALUGUEL", "LOCAÇÃO": "ALUGUEL",
    "ALUGAR": "ALUGUEL",
}

# Valores que denunciam coluna trocada: são TIPO aparecendo no campo OFERTA.
VOCAB_TIPO = {
    "APARTAMENTO", "CASA", "CASA CONDOMINIO", "CASA DE CONDOMINIO", "LOTE",
    "SALA", "LOJA", "PREDIO", "PRÉDIO", "KITNET", "GALPAO", "GALPÃO", "RURAL",
    "PONTO COMERCIAL", "HOTEL-FLAT", "FLAT", "GARAGEM", "LOTEAMENTO",
    "TERRENO", "COBERTURA", "SOBRADO",
}

COLUNAS_CANONICAS = [
    "portal", "id_anuncio", "codigo", "link", "creci", "anunciante",
    "oferta", "tipo", "area_util", "bairro", "cidade", "preco", "valor_m2",
    "quartos", "vagas", "latitude", "longitude", "quadra", "data_coleta",
]
COLUNAS_NUMERICAS = ["area_util", "preco", "valor_m2", "quartos", "vagas",
                     "latitude", "longitude"]


# ============================================================
# Conexão
# ============================================================

def conectar():
    faltando = [k for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD")
                if not os.getenv(k)]
    if faltando:
        raise RuntimeError(
            "Variáveis de ambiente ausentes: " + ", ".join(faltando) +
            ". Defina no .env antes de rodar."
        )
    return psycopg2.connect(
        host=os.getenv("PGHOST"), port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE"), user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"), connect_timeout=20,
    )


def raiz_dados() -> Path:
    return Path(os.getenv("DATA_DIR", "dados")).expanduser()


# ============================================================
# Descoberta de arquivos
# ============================================================

def descobrir(pasta: Optional[Path] = None) -> list[dict]:
    """
    Varre dados/coletas/<portal>/*.csv e deduz portal e data.

    Portal vem da PASTA, não do nome do arquivo nem de constante no código —
    é o que impede o `PORTAL = "df"` fixo de rotular coleta do Wimóveis como
    DF Imóveis.
    """
    base = (pasta or raiz_dados() / "coletas")
    if not base.exists():
        print(f"[AVISO] pasta inexistente: {base}")
        return []

    achados = []
    for arq in sorted(base.rglob("*.csv")):
        portal = arq.parent.name.strip().lower()
        if portal not in PORTAIS_CONHECIDOS:
            print(f"[AVISO] ignorado (pasta não é um portal conhecido): {arq}")
            continue
        m = RE_DATA.search(arq.name)
        if not m:
            print(f"[AVISO] ignorado (sem data AAAA-MM-DD no nome): {arq}")
            continue
        achados.append({"arquivo": arq, "portal": portal, "data_coleta": m.group(1)})
    return achados


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


# ============================================================
# Normalização
# ============================================================

def detectar_layout(df: pd.DataFrame) -> str:
    """`tipo_imovel` presente = layout novo, em que `tipo` carrega a oferta."""
    return "novo" if "tipo_imovel" in df.columns else "antigo"


def _limpar_texto(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.upper()
             .replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA, "NONE": pd.NA}))


def normalizar(df: pd.DataFrame, portal: str, data_coleta: str) -> pd.DataFrame:
    """CSV de qualquer layout -> dataframe canônico."""
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                    .str.replace("﻿", "", regex=False)
                    .str.replace(" ", "_", regex=False))

    layout = detectar_layout(df)
    out = pd.DataFrame(index=df.index)

    if layout == "novo":
        # `tipo` do CSV novo é a OFERTA; `tipo_imovel` é o tipo.
        out["oferta"] = _limpar_texto(df.get("tipo", pd.Series(dtype="object")))
        out["tipo"] = _limpar_texto(df.get("tipo_imovel", pd.Series(dtype="object")))
    else:
        out["oferta"] = _limpar_texto(df.get("oferta", pd.Series(dtype="object")))
        out["tipo"] = _limpar_texto(df.get("tipo", pd.Series(dtype="object")))

    # Rede de segurança: se ainda assim vier trocado (CSV antigo já corrompido),
    # desfaz a troca linha a linha em vez de deixar entrar errado.
    trocado = out["oferta"].isin(VOCAB_TIPO) & out["tipo"].isin(MAPA_OFERTA.keys())
    if trocado.any():
        print(f"[INFO] {int(trocado.sum())} linhas com oferta/tipo invertidos — corrigido")
        o, t = out.loc[trocado, "oferta"].copy(), out.loc[trocado, "tipo"].copy()
        out.loc[trocado, "oferta"], out.loc[trocado, "tipo"] = t, o

    out["oferta"] = out["oferta"].map(lambda v: MAPA_OFERTA.get(v, v) if pd.notna(v) else v)

    # id do anúncio: chave estável do portal. O `codigo` da imobiliária falha
    # (o scraperWI extrai por regex da página de detalhe e pode vir vazio).
    id_anuncio = _limpar_texto(df.get("id", pd.Series(dtype="object")))
    codigo = _limpar_texto(df.get("codigo", pd.Series(dtype="object")))
    if id_anuncio.isna().all() and "link" in df.columns:
        # Sem coluna id: extrai o id numérico do fim da URL.
        id_anuncio = (df["link"].astype("string")
                      .str.extract(r"(\d{4,})(?:\.html)?/?$", expand=False)
                      .str.upper())
    out["id_anuncio"] = id_anuncio.fillna(codigo)
    out["codigo"] = codigo.fillna(id_anuncio)

    for c in ["link", "creci", "anunciante", "bairro", "cidade", "quadra"]:
        out[c] = _limpar_texto(df[c]) if c in df.columns else pd.NA
    out["link"] = df["link"].astype("string").str.strip() if "link" in df.columns else pd.NA

    for c in COLUNAS_NUMERICAS:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.NA

    # valor_m2 derivado quando o portal não manda (o Wimóveis costuma omitir).
    faltando_m2 = out["valor_m2"].isna() & out["preco"].notna() & (out["area_util"] > 0)
    out.loc[faltando_m2, "valor_m2"] = out.loc[faltando_m2, "preco"] / out.loc[faltando_m2, "area_util"]

    # Data: a do arquivo manda; a coluna `data` do CSV serve de conferência.
    out["data_coleta"] = pd.to_datetime(data_coleta).date()
    if "data" in df.columns:
        do_csv = pd.to_datetime(df["data"], errors="coerce").dt.date.dropna().unique()
        divergentes = [d for d in do_csv if str(d) != data_coleta]
        if divergentes:
            print(f"[AVISO] coluna `data` do CSV diverge do nome do arquivo "
                  f"({data_coleta}): {divergentes[:3]}")

    out["portal"] = portal.strip().lower()
    return out[COLUNAS_CANONICAS]


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descarta o que não serve e reporta. Melhor perder a linha na entrada do que
    descobrir depois que a mediana do bairro mudou por causa de lixo.
    """
    antes = len(df)
    relatorio = []

    sem_chave = df["id_anuncio"].isna() & df["codigo"].isna()
    if sem_chave.any():
        relatorio.append(f"sem id e sem codigo: {int(sem_chave.sum())}")
    sem_oferta = ~df["oferta"].isin(["VENDA", "ALUGUEL"])
    if sem_oferta.any():
        relatorio.append(f"oferta fora de VENDA/ALUGUEL: {int(sem_oferta.sum())}")
    sem_preco = df["preco"].isna() | (df["preco"] <= 0)
    if sem_preco.any():
        relatorio.append(f"preço ausente ou <= 0: {int(sem_preco.sum())}")

    ruim = sem_chave | sem_oferta | sem_preco
    df = df[~ruim].copy()

    # Duplicata dentro do próprio arquivo: o portal repete o card entre páginas.
    dups = df.duplicated(subset=["portal", "id_anuncio", "data_coleta"], keep="first")
    if dups.any():
        relatorio.append(f"duplicadas no arquivo: {int(dups.sum())}")
        df = df[~dups].copy()

    if relatorio:
        print("[VALIDAÇÃO] " + " | ".join(relatorio))
    print(f"[VALIDAÇÃO] {antes} linhas -> {len(df)} aproveitadas "
          f"({len(df)/max(antes,1):.1%})")
    return df


# ============================================================
# Banco
# ============================================================

def preparar_banco(conn) -> None:
    """
    DDL idempotente. Não cria índice único sobre o histórico: a base atual tem
    ~35 mil chaves (portal, codigo, data_coleta) duplicadas e a criação
    falharia. Ver `--relatorio-duplicatas`.
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            create schema if not exists {SCHEMA_CTRL};

            -- id do anúncio no portal: chave estável, independente do código
            -- da imobiliária (que o Wimóveis nem sempre expõe).
            alter table {TABELA} add column if not exists id_anuncio text;

            create index if not exists idx_{TABELA}_portal_chave
                on {TABELA} (portal, id_anuncio, data_coleta);

            -- Controle de carga: arquivo já processado não volta ao banco.
            create table if not exists {SCHEMA_CTRL}.{TBL_CARGAS} (
              arquivo      text not null,
              portal       text not null,
              data_coleta  date not null,
              sha256       text not null,
              linhas_lidas int  not null,
              linhas_ok    int  not null,
              carregado_em timestamp not null default now(),
              primary key (portal, data_coleta, sha256)
            );
        """)
    conn.commit()


def ja_carregado(conn, portal: str, data_coleta: str, digest: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            f"select 1 from {SCHEMA_CTRL}.{TBL_CARGAS} "
            f"where portal=%s and data_coleta=%s and sha256=%s",
            (portal, data_coleta, digest))
        return cur.fetchone() is not None


def inserir(conn, df: pd.DataFrame, page_size: int = 2000) -> int:
    if df.empty:
        return 0
    cols = COLUNAS_CANONICAS
    bloco = df[cols].astype(object)
    bloco = bloco.where(bloco.notna(), None)
    registros = list(bloco.itertuples(index=False, name=None))

    q = psql.SQL("insert into {tbl} ({cols}) values %s").format(
        tbl=psql.Identifier(TABELA),
        cols=psql.SQL(", ").join(map(psql.Identifier, cols)),
    )
    with conn.cursor() as cur:
        execute_values(cur, q.as_string(conn), registros, page_size=page_size)
    return len(registros)


def apagar_carga(conn, portal: str, data_coleta: str) -> int:
    """
    Apaga o que já existe para (portal, data_coleta) antes de recarregar.

    Serve para consertar carga malformada: as coletas de agosto/2026 entraram
    com `oferta` vazia porque o layout do CSV mudou, e o tipo do imóvel se
    perdeu no caminho. Reingerir sem apagar duplicaria o dia inteiro.

    Destrutivo por definição — só roda com --substituir explícito.
    """
    with conn.cursor() as cur:
        cur.execute(f"delete from {TABELA} where portal = %s and data_coleta = %s",
                    (portal, data_coleta))
        apagadas = cur.rowcount
        cur.execute(
            f"delete from {SCHEMA_CTRL}.{TBL_CARGAS} where portal=%s and data_coleta=%s",
            (portal, data_coleta))
    return apagadas


def contar_existentes(conn, portal: str, data_coleta: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {TABELA} where portal=%s and data_coleta=%s",
                    (portal, data_coleta))
        return cur.fetchone()[0]


def registrar_carga(conn, arquivo: Path, portal: str, data_coleta: str,
                    digest: str, lidas: int, ok: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"insert into {SCHEMA_CTRL}.{TBL_CARGAS} "
            f"(arquivo, portal, data_coleta, sha256, linhas_lidas, linhas_ok) "
            f"values (%s,%s,%s,%s,%s,%s) on conflict do nothing",
            (str(arquivo), portal, data_coleta, digest, lidas, ok))


def relatorio_duplicatas(conn, dias: int = 3650) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            select count(*) as chaves, coalesce(sum(c),0) as linhas, coalesce(max(c),0) as pior
            from (
              select portal, trim(codigo) as codigo, data_coleta, count(*) as c
              from {TABELA}
              where data_coleta >= current_date - %s
                and coalesce(trim(codigo),'') <> ''
              group by 1,2,3 having count(*) > 1
            ) t
        """, (dias,))
        chaves, linhas, pior = cur.fetchone()
    print(f"[DUPLICATAS] {chaves} chaves (portal, codigo, data_coleta) repetidas, "
          f"{linhas} linhas, pior caso {pior}x")
    if chaves:
        print("  Para criar o índice único é preciso remover as repetições antes.")
        print("  A consulta abaixo mostra o que seria apagado (NÃO executa nada):")
        print(f"""
    with ranqueado as (
      select ctid, row_number() over (
               partition by portal, trim(codigo), data_coleta order by ctid
             ) as rn
      from {TABELA} where coalesce(trim(codigo),'') <> ''
    )
    select count(*) from ranqueado where rn > 1;   -- linhas excedentes
""")


# ============================================================
# Fluxo
# ============================================================

def processar(cargas: Iterable[dict], dry_run: bool = False,
              substituir: bool = False) -> None:
    cargas = list(cargas)
    if not cargas:
        print("Nada a carregar.")
        return

    # dry-run nao abre conexao: confere o parse sem tocar no banco.
    conn = None if dry_run else conectar()
    try:
        if conn is not None:
            preparar_banco(conn)

        for c in cargas:
            arq, portal, dt = Path(c["arquivo"]), c["portal"], c["data_coleta"]
            print(f"\n=== {portal.upper()} / {dt} / {arq.name} ===")
            digest = sha256(arq)

            if conn is not None and substituir:
                existentes = contar_existentes(conn, portal, dt)
                if existentes:
                    print(f"  --substituir: {existentes} linhas já gravadas "
                          f"para {portal}/{dt} serão APAGADAS antes da recarga")
            elif not dry_run and ja_carregado(conn, portal, dt, digest):
                print("  já carregado (mesmo conteúdo) — pulado")
                continue

            # Coleta que falhou deixa arquivo vazio (o 2026-08-24 é um caso
            # real). Pular com aviso é melhor que derrubar o lote inteiro.
            if arq.stat().st_size == 0:
                print("  arquivo vazio (coleta falhou?) — pulado")
                continue
            try:
                bruto = pd.read_csv(arq, dtype=str, encoding="utf-8-sig",
                                    keep_default_na=False, na_values=[""])
            except pd.errors.EmptyDataError:
                print("  arquivo sem cabeçalho utilizável — pulado")
                continue
            if bruto.empty:
                print("  arquivo sem linhas de dados — pulado")
                continue
            print(f"  layout: {detectar_layout(bruto)}  |  {len(bruto)} linhas")
            df = validar(normalizar(bruto, portal, dt))

            if dry_run:
                print("  [dry-run] nada gravado"
                      + (f" (--substituir apagaria as linhas de {portal}/{dt} "
                         f"antes de inserir)" if substituir else ""))
                print(df[["portal", "oferta", "tipo"]].value_counts().head(8).to_string())
                continue

            if substituir:
                apagadas = apagar_carga(conn, portal, dt)
                print(f"  {apagadas} linhas antigas apagadas.")
            n = inserir(conn, df)
            registrar_carga(conn, arq, portal, dt, digest, len(bruto), n)
            conn.commit()
            print(f"  {n} linhas inseridas.")
    finally:
        if conn is not None:
            conn.close()

    print("\nIngestão concluída.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingestão de coletas na tabela imoveis")
    ap.add_argument("--pasta", type=Path, help="raiz das coletas (padrão: $DATA_DIR/coletas)")
    ap.add_argument("--arquivo", type=Path, help="carrega um CSV específico")
    ap.add_argument("--portal", help="portal do --arquivo (df|wi)")
    ap.add_argument("--data", help="data AAAA-MM-DD do --arquivo (padrão: do nome)")
    ap.add_argument("--dry-run", action="store_true", help="mostra sem gravar")
    ap.add_argument("--substituir", action="store_true",
                    help="APAGA as linhas já gravadas de (portal, data) antes de "
                         "inserir. Use para recarregar coleta malformada.")
    ap.add_argument("--relatorio-duplicatas", action="store_true")
    args = ap.parse_args()

    if args.relatorio_duplicatas:
        with closing(conectar()) as conn:
            relatorio_duplicatas(conn)
        return

    if args.arquivo:
        portal = (args.portal or args.arquivo.parent.name).strip().lower()
        if portal not in PORTAIS_CONHECIDOS:
            raise SystemExit(f"Informe --portal ({'|'.join(sorted(PORTAIS_CONHECIDOS))})")
        m = RE_DATA.search(args.arquivo.name)
        dt = args.data or (m.group(1) if m else None)
        if not dt:
            raise SystemExit("Sem data no nome do arquivo: informe --data AAAA-MM-DD")
        cargas = [{"arquivo": args.arquivo, "portal": portal, "data_coleta": dt}]
    else:
        cargas = descobrir(args.pasta)

    processar(cargas, dry_run=args.dry_run, substituir=args.substituir)


if __name__ == "__main__":
    main()
