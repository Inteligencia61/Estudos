# -*- coding: utf-8 -*-
"""
Histórico de listagem e métricas de mercado (Fase 2).

O `acionador.py` responde "quanto pedem": mediana de R$/m² do que está no ar.
Este módulo responde as outras três perguntas que um estudo de mercado precisa
responder e que o dado já coletado permite, sem coleta nova:

    - quanto tempo o imóvel leva para sair do portal   -> DOM
    - quanto do preço pedido o mercado devolve          -> redução de preço
    - qual a velocidade de escoamento do estoque        -> absorção

Tudo sai do painel semanal que já existe em `imoveis`: o mesmo `codigo`
aparece em várias `data_coleta`, então dá para reconstruir a vida de cada
anúncio (quando entrou, quanto pediu, quantas vezes baixou, quando sumiu).

LIMITE HONESTO: "sumir do portal" NÃO é "vendeu". Pode ser venda, contrato
expirado, anúncio pausado ou o mesmo imóvel republicado com outro código.
Enquanto o Estoque (captação/saída da 61) não for cruzado, isto é uma PROXY de
absorção — boa para comparar bairros entre si e acompanhar tendência, não para
afirmar volume de vendas em número absoluto.

Tabelas geradas:
    analytics.listagem_historico  -> uma linha por anúncio (código x oferta)
    analytics.mercado_metricas    -> uma linha por bairro x tipo x oferta x mês

Uso:
    python listagem_historico.py                 # reconstrói tudo (12 meses)
    python listagem_historico.py --meses 24
"""
from __future__ import annotations

import argparse
import os
from contextlib import closing
from datetime import date
from typing import Optional

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql as psql

load_dotenv()


# ============================================================
# Conexão
# ============================================================

def conectar():
    """Credenciais só do ambiente — nada embutido no código."""
    faltando = [k for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD")
                if not os.getenv(k)]
    if faltando:
        raise RuntimeError(
            "Variáveis de ambiente ausentes: " + ", ".join(faltando) +
            ". Defina-as no .env antes de rodar."
        )
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )


class HistoricoListagens:
    """
    Reconstrói a vida de cada anúncio e deriva as métricas de mercado.

        h = HistoricoListagens(meses=12)
        h.construir()
        print(h.resumo("LAGO SUL", "CASA", "VENDA"))
    """

    def __init__(
        self,
        meses: int = 12,
        schema: str = "analytics",
        tabela_fonte: str = "imoveis",
        tbl_historico: str = "listagem_historico",
        tbl_mercado: str = "mercado_metricas",
        # Uma coleta semanal atrasada não pode marcar o mercado inteiro como
        # "saiu": só é considerado encerrado o anúncio ausente há mais dias
        # que isto, contados da última coleta existente na base.
        dias_tolerancia_saida: int = 14,
        # Variação mínima para contar como redução/aumento de preço, filtrando
        # correção de digitação e reajuste de centavos.
        limiar_variacao: float = 0.005,
        ofertas: tuple = ("VENDA", "ALUGUEL"),
    ):
        self.meses = int(meses)
        self.schema = schema
        self.tabela_fonte = tabela_fonte
        self.tbl_historico = tbl_historico
        self.tbl_mercado = tbl_mercado
        self.dias_tolerancia_saida = int(dias_tolerancia_saida)
        self.limiar_variacao = float(limiar_variacao)
        self.ofertas = tuple(ofertas)

    # ----------------------------------------------------------
    # DDL
    # ----------------------------------------------------------

    def _ddl(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(psql.SQL("create schema if not exists {}").format(
                psql.Identifier(self.schema)))
            cur.execute(psql.SQL("""
                create index if not exists idx_{idx}_cod_data
                    on {fonte} (codigo, data_coleta)
            """).format(
                idx=psql.SQL(self.tabela_fonte),
                fonte=psql.Identifier(self.tabela_fonte),
            ))

    # ----------------------------------------------------------
    # 1) Histórico por anúncio
    # ----------------------------------------------------------

    def construir_historico(self, conn) -> int:
        """
        Uma linha por (código, oferta) com a vida inteira do anúncio.

        `preco_inicial` é o primeiro preço visto na janela, não
        necessariamente o de lançamento: anúncio que já estava no ar quando a
        coleta começou entra censurado à esquerda. Por isso `censurado_esq`.
        """
        q = psql.SQL("""
        drop table if exists {schema}.{tbl};
        create table {schema}.{tbl} as
        with limites as (
            select max(data_coleta)::date as ultima_coleta,
                   (max(data_coleta) - make_interval(months => %(meses)s))::date as inicio
            from {fonte}
        ),
        base as (
            -- um registro por anúncio por coleta (a fonte repete o card)
            select distinct on (i.codigo, i.oferta, i.data_coleta)
                trim(i.codigo)::text            as codigo,
                upper(trim(i.oferta))::text     as oferta,
                upper(trim(i.bairro))::text     as bairro,
                upper(trim(i.cidade))::text     as cidade,
                upper(trim(i.tipo))::text       as tipo,
                upper(trim(i.quadra))::text     as quadra,
                i.preco::double precision       as preco,
                i.area_util::double precision   as area_util,
                i.quartos::double precision     as quartos,
                i.vagas::double precision       as vagas,
                i.data_coleta::date             as data_coleta
            from {fonte} i, limites l
            where i.data_coleta >= l.inicio
              and i.preco is not null and i.preco > 0
              and i.area_util is not null and i.area_util > 0
              and coalesce(trim(i.codigo), '') <> ''
              and upper(trim(i.oferta)) = any(%(ofertas)s)
            order by i.codigo, i.oferta, i.data_coleta, i.preco asc
        ),
        seq as (
            select b.*,
                lag(preco) over (partition by codigo, oferta order by data_coleta) as preco_ant
            from base b
        ),
        mov as (
            select codigo, oferta,
                count(*) filter (
                    where preco_ant is not null and preco < preco_ant * (1 - %(limiar)s)
                ) as n_reducoes,
                count(*) filter (
                    where preco_ant is not null and preco > preco_ant * (1 + %(limiar)s)
                ) as n_aumentos
            from seq group by 1, 2
        ),
        vida as (
            select
                codigo, oferta,
                mode() within group (order by bairro)  as bairro,
                mode() within group (order by cidade)  as cidade,
                mode() within group (order by tipo)    as tipo,
                mode() within group (order by quadra)  as quadra,
                min(data_coleta)                       as primeira_vez,
                max(data_coleta)                       as ultima_vez,
                count(*)                               as n_coletas,
                (array_agg(preco order by data_coleta asc))[1]  as preco_inicial,
                (array_agg(preco order by data_coleta desc))[1] as preco_final,
                min(preco)                             as preco_min,
                max(preco)                             as preco_max,
                avg(area_util)                         as area_util,
                max(quartos)                           as quartos,
                max(vagas)                             as vagas
            from base group by codigo, oferta
        )
        select
            v.codigo, v.oferta, v.bairro, v.cidade, v.tipo, v.quadra,
            v.primeira_vez, v.ultima_vez, v.n_coletas,
            v.preco_inicial, v.preco_final, v.preco_min, v.preco_max,
            v.area_util, v.quartos, v.vagas,
            (v.preco_final / nullif(v.area_util, 0))          as valor_m2_final,
            coalesce(m.n_reducoes, 0)                         as n_reducoes,
            coalesce(m.n_aumentos, 0)                         as n_aumentos,
            round((v.preco_final / nullif(v.preco_inicial, 0) - 1)::numeric * 100, 2)
                                                              as variacao_preco_pct,
            (v.ultima_vez - v.primeira_vez)                   as dias_no_ar,
            -- anúncio visto na última janela de coleta continua vivo: o DOM
            -- dele ainda não terminou e não pode entrar na mediana de saída
            (v.ultima_vez >= (l.ultima_coleta - %(tolerancia)s))         as ativo,
            (v.primeira_vez <= l.inicio)                                 as censurado_esq,
            to_char(v.primeira_vez, 'YYYY-MM')                as mes_entrada,
            to_char(v.ultima_vez, 'YYYY-MM')                  as mes_saida,
            l.ultima_coleta                                   as ref_ultima_coleta,
            now()                                             as gerado_em
        from vida v
        left join mov m on m.codigo = v.codigo and m.oferta = v.oferta
        cross join limites l;

        create index if not exists {idx_escopo}
            on {schema}.{tbl} (bairro, tipo, oferta);
        create index if not exists {idx_datas}
            on {schema}.{tbl} (primeira_vez, ultima_vez);
        """).format(
            schema=psql.Identifier(self.schema),
            tbl=psql.Identifier(self.tbl_historico),
            fonte=psql.Identifier(self.tabela_fonte),
            idx_escopo=psql.Identifier(f"idx_{self.tbl_historico}_escopo"),
            idx_datas=psql.Identifier(f"idx_{self.tbl_historico}_datas"),
        )

        with conn.cursor() as cur:
            cur.execute(q, {
                "meses": self.meses,
                "ofertas": list(self.ofertas),
                "limiar": self.limiar_variacao,
                "tolerancia": self.dias_tolerancia_saida,
            })
            cur.execute(psql.SQL("select count(*) from {}.{}").format(
                psql.Identifier(self.schema), psql.Identifier(self.tbl_historico)))
            n = cur.fetchone()[0]
        conn.commit()
        print(f"[OK] {self.schema}.{self.tbl_historico}: {n} anúncios reconstruídos.")
        return n

    # ----------------------------------------------------------
    # 2) Métricas de mercado por mês
    # ----------------------------------------------------------

    def construir_mercado(self, conn) -> int:
        """
        Por bairro x tipo x oferta x mês:

            estoque_ativo   anúncios no ar em algum momento do mês
            entradas        anúncios que apareceram no mês
            saidas          anúncios cuja última aparição foi no mês (e não
                            estão ativos) — proxy de escoamento, ver docstring
                            do módulo
            absorcao_pct    saidas / estoque_ativo
            meses_estoque   estoque_ativo / saidas  ("em quantos meses o
                            estoque atual escoa no ritmo do mês")
            dom_mediano     mediana de dias_no_ar entre os que saíram
            pct_com_reducao % do estoque do mês que já baixou o preço ao menos
                            uma vez
            desconto_mediano_pct  mediana da variação de preço entre quem
                            baixou — o número que sustenta conversa de
                            precificação com proprietário
        """
        q = psql.SQL("""
        drop table if exists {schema}.{tbl};
        create table {schema}.{tbl} as
        with h as (
            select * from {schema}.{hist}
        ),
        meses as (
            select to_char(gs, 'YYYY-MM') as mes_ref,
                   gs::date               as mes_inicio,
                   (gs + interval '1 month - 1 day')::date as mes_fim
            from generate_series(
                (select date_trunc('month', min(primeira_vez)) from h),
                (select date_trunc('month', max(ultima_vez))  from h),
                interval '1 month'
            ) gs
        ),
        cruz as (
            select m.mes_ref, m.mes_inicio, m.mes_fim,
                   h.bairro, h.tipo, h.oferta,
                   h.dias_no_ar, h.ativo, h.n_reducoes, h.variacao_preco_pct,
                   h.valor_m2_final,
                   (h.primeira_vez >= m.mes_inicio and h.primeira_vez <= m.mes_fim) as entrou,
                   (h.ultima_vez  >= m.mes_inicio and h.ultima_vez  <= m.mes_fim
                    and not h.ativo)                                               as saiu
            from meses m
            join h
              on h.primeira_vez <= m.mes_fim
             and h.ultima_vez   >= m.mes_inicio
        )
        select
            bairro, tipo, oferta, mes_ref, mes_inicio, mes_fim,
            count(*)                                          as estoque_ativo,
            count(*) filter (where entrou)                    as entradas,
            count(*) filter (where saiu)                      as saidas,
            round((count(*) filter (where saiu))::numeric
                  / nullif(count(*), 0) * 100, 2)             as absorcao_pct,
            round(count(*)::numeric
                  / nullif(count(*) filter (where saiu), 0), 1) as meses_estoque,
            percentile_cont(0.5) within group (order by dias_no_ar)
                filter (where saiu)                           as dom_mediano,
            percentile_cont(0.5) within group (order by valor_m2_final)
                                                              as m2_mediana_estoque,
            round((count(*) filter (where n_reducoes > 0))::numeric
                  / nullif(count(*), 0) * 100, 2)             as pct_com_reducao,
            round((percentile_cont(0.5) within group (order by variacao_preco_pct)
                   filter (where n_reducoes > 0))::numeric, 2) as desconto_mediano_pct,
            now()                                             as gerado_em
        from cruz
        group by bairro, tipo, oferta, mes_ref, mes_inicio, mes_fim;

        create index if not exists {idx_escopo}
            on {schema}.{tbl} (bairro, tipo, oferta, mes_ref);
        """).format(
            schema=psql.Identifier(self.schema),
            tbl=psql.Identifier(self.tbl_mercado),
            hist=psql.Identifier(self.tbl_historico),
            idx_escopo=psql.Identifier(f"idx_{self.tbl_mercado}_escopo"),
        )

        with conn.cursor() as cur:
            cur.execute(q)
            cur.execute(psql.SQL("select count(*) from {}.{}").format(
                psql.Identifier(self.schema), psql.Identifier(self.tbl_mercado)))
            n = cur.fetchone()[0]
        conn.commit()
        print(f"[OK] {self.schema}.{self.tbl_mercado}: {n} linhas bairro/tipo/oferta/mês.")
        return n

    # ----------------------------------------------------------
    # API
    # ----------------------------------------------------------

    def construir(self) -> None:
        with closing(conectar()) as conn:
            self._ddl(conn)
            conn.commit()
            self.construir_historico(conn)
            self.construir_mercado(conn)

    def resumo(self, bairro: str, tipo: str, oferta: str = "VENDA",
               ultimos: int = 6) -> pd.DataFrame:
        """Série mensal de mercado para um escopo, pronta para gráfico/relatório."""
        q = psql.SQL("""
            select mes_ref, estoque_ativo, entradas, saidas, absorcao_pct,
                   meses_estoque, dom_mediano, m2_mediana_estoque,
                   pct_com_reducao, desconto_mediano_pct
            from {schema}.{tbl}
            where bairro = %s and tipo = %s and oferta = %s
            order by mes_ref desc
            limit %s
        """).format(schema=psql.Identifier(self.schema),
                    tbl=psql.Identifier(self.tbl_mercado))
        with closing(conectar()) as conn:
            df = pd.read_sql(q.as_string(conn), conn,
                             params=(bairro.upper(), tipo.upper(), oferta.upper(), ultimos))
        return df.sort_values("mes_ref")

    def ranking_absorcao(self, tipo: str = "CASA", oferta: str = "VENDA",
                         mes_ref: Optional[str] = None,
                         min_estoque: int = 30) -> pd.DataFrame:
        """
        Bairros ordenados por velocidade de escoamento no mês.

        É a leitura que o comercial usa: onde o estoque gira rápido, captar
        no preço de mercado; onde `meses_estoque` está alto, a conversa com o
        proprietário começa pelo prazo, não pelo valor.
        """
        filtro_mes = psql.SQL("and mes_ref = %s") if mes_ref else psql.SQL(
            "and mes_ref = (select max(mes_ref) from {schema}.{tbl})"
        ).format(schema=psql.Identifier(self.schema),
                 tbl=psql.Identifier(self.tbl_mercado))

        q = psql.SQL("""
            select bairro, mes_ref, estoque_ativo, saidas, absorcao_pct,
                   meses_estoque, dom_mediano, pct_com_reducao,
                   desconto_mediano_pct, m2_mediana_estoque
            from {schema}.{tbl}
            where tipo = %s and oferta = %s and estoque_ativo >= %s
            {filtro_mes}
            order by absorcao_pct desc nulls last
        """).format(schema=psql.Identifier(self.schema),
                    tbl=psql.Identifier(self.tbl_mercado),
                    filtro_mes=filtro_mes)

        params = [tipo.upper(), oferta.upper(), min_estoque]
        if mes_ref:
            params.append(mes_ref)
        with closing(conectar()) as conn:
            return pd.read_sql(q.as_string(conn), conn, params=tuple(params))


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Histórico de listagem e métricas de mercado")
    ap.add_argument("--meses", type=int, default=12,
                    help="janela de reconstrução, em meses (padrão 12)")
    ap.add_argument("--resumo", nargs=2, metavar=("BAIRRO", "TIPO"),
                    help="imprime a série mensal do escopo depois de construir")
    ap.add_argument("--oferta", default="VENDA")
    ap.add_argument("--somente-resumo", action="store_true",
                    help="não reconstrói as tabelas, só lê")
    args = ap.parse_args()

    h = HistoricoListagens(meses=args.meses)
    if not args.somente_resumo:
        h.construir()

    if args.resumo:
        bairro, tipo = args.resumo
        print(f"\n== {bairro.upper()} / {tipo.upper()} / {args.oferta.upper()} ==")
        print(h.resumo(bairro, tipo, args.oferta).to_string(index=False))
        print(f"\n== Ranking de absorção — {tipo.upper()} / {args.oferta.upper()} ==")
        print(h.ranking_absorcao(tipo, args.oferta).to_string(index=False))
