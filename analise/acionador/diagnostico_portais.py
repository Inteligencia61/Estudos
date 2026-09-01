# -*- coding: utf-8 -*-
"""
Diagnóstico somente-leitura da tabela `imoveis`.

Com mais de um portal alimentando a mesma tabela, três coisas quebram em
silêncio: o rótulo `portal` errado, `codigo` colidindo entre portais e
vocabulário divergente em `oferta`/`tipo`. Nenhuma delas dá erro — só muda o
número do estudo.

Rode depois de cada carga mensal:

    python analise/acionador/diagnostico_portais.py

Não escreve nada. Só SELECT.
"""
from __future__ import annotations

import os
from contextlib import closing

import psycopg2
from dotenv import load_dotenv

load_dotenv()

TABELA = "imoveis"
JANELA_DIAS = 90          # janela das checagens caras
CHAVE = "coalesce(nullif(trim(codigo), ''), '')"


def conectar():
    faltando = [k for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD")
                if not os.getenv(k)]
    if faltando:
        raise RuntimeError("Variáveis ausentes: " + ", ".join(faltando))
    return psycopg2.connect(
        host=os.getenv("PGHOST"), port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE"), user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"), connect_timeout=20,
    )


def secao(titulo: str) -> None:
    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)


def tabela(cur, titulo: str, sql: str, params=None) -> list:
    secao(titulo)
    cur.execute(sql, params or ())
    linhas = cur.fetchall()
    if not linhas:
        print("  (vazio)")
        return []
    cols = [d[0] for d in cur.description]
    larg = [max(len(str(c)), *(len(str(l[i])) for l in linhas))
            for i, c in enumerate(cols)]
    print("  " + "  ".join(str(c).ljust(larg[i]) for i, c in enumerate(cols)))
    print("  " + "  ".join("-" * larg[i] for i in range(len(cols))))
    for l in linhas:
        print("  " + "  ".join(str(v).ljust(larg[i]) for i, v in enumerate(l)))
    return linhas


def main() -> None:
    with closing(conectar()) as conn:
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("set statement_timeout = '180s'")

            tabela(cur, "1. PORTAIS — volume e período", f"""
                select coalesce(portal, '(null)')      as portal,
                       count(*)                        as linhas,
                       count(distinct codigo)          as codigos,
                       count(distinct data_coleta)     as coletas,
                       min(data_coleta)                as desde,
                       max(data_coleta)                as ate
                from {TABELA}
                group by 1 order by 2 desc
            """)

            tabela(cur, "2. VOCABULÁRIO de `oferta` por portal", f"""
                select coalesce(portal,'(null)') as portal,
                       coalesce(nullif(upper(trim(oferta)),''),'(vazio)') as oferta,
                       count(*) as linhas
                from {TABELA}
                where data_coleta >= current_date - {JANELA_DIAS}
                group by 1,2 order by 1, 3 desc
            """)

            tabela(cur, "3. VOCABULÁRIO de `tipo` por portal (top 12)", f"""
                select portal, tipo, linhas from (
                  select coalesce(portal,'(null)') as portal,
                         coalesce(nullif(upper(trim(tipo)),''),'(vazio)') as tipo,
                         count(*) as linhas,
                         row_number() over (partition by coalesce(portal,'(null)')
                                            order by count(*) desc) as rn
                  from {TABELA}
                  where data_coleta >= current_date - {JANELA_DIAS}
                  group by 1,2
                ) t where rn <= 12 order by portal, linhas desc
            """)

            tabela(cur, "4. PREENCHIMENTO por portal (últimos %d dias, %% vazio)"
                   % JANELA_DIAS, f"""
                select coalesce(portal,'(null)') as portal,
                       count(*) as linhas,
                       round(100.0*count(*) filter (where coalesce(trim(codigo),'')='')
                             /nullif(count(*),0),1) as codigo_vazio,
                       round(100.0*count(*) filter (where latitude is null)
                             /nullif(count(*),0),1) as lat_null,
                       round(100.0*count(*) filter (where area_util is null or area_util<=0)
                             /nullif(count(*),0),1) as area_ruim,
                       round(100.0*count(*) filter (where preco is null or preco<=0)
                             /nullif(count(*),0),1) as preco_ruim,
                       round(100.0*count(*) filter (where coalesce(trim(quadra),'')='')
                             /nullif(count(*),0),1) as quadra_vazia,
                       round(100.0*count(*) filter (where coalesce(trim(anunciante),'')='')
                             /nullif(count(*),0),1) as anunc_vazio
                from {TABELA}
                where data_coleta >= current_date - {JANELA_DIAS}
                group by 1 order by 2 desc
            """)

            tabela(cur, "5. COLISÃO de `codigo` entre portais", f"""
                with por_codigo as (
                  select trim(codigo) as codigo,
                         count(distinct portal) as n_portais,
                         string_agg(distinct portal, ',') as portais
                  from {TABELA}
                  where coalesce(trim(codigo),'') <> ''
                  group by 1
                )
                select n_portais, portais, count(*) as codigos_afetados
                from por_codigo where n_portais > 1
                group by 1,2 order by 3 desc limit 10
            """)

            tabela(cur, "6. DUPLICATA de (portal, codigo, data_coleta)", f"""
                select count(*) as chaves_duplicadas,
                       sum(c) as linhas_envolvidas,
                       max(c) as pior_caso
                from (
                  select portal, trim(codigo) as codigo, data_coleta, count(*) as c
                  from {TABELA}
                  where data_coleta >= current_date - {JANELA_DIAS}
                    and coalesce(trim(codigo),'') <> ''
                  group by 1,2,3 having count(*) > 1
                ) t
            """)

            tabela(cur, "7. CADÊNCIA de coleta (últimas 12 datas)", f"""
                select data_coleta, coalesce(portal,'(null)') as portal, count(*) as linhas
                from {TABELA}
                where data_coleta >= current_date - 120
                group by 1,2 order by 1 desc, 3 desc limit 24
            """)

            tabela(cur, "8. O QUE O ACIONADOR ENXERGA hoje", f"""
                select coalesce(portal,'(null)') as portal,
                       count(*) as linhas_janela,
                       count(*) filter (
                         where upper(trim(oferta)) in ('VENDA','ALUGUEL','PUBLICADO')
                       ) as oferta_valida,
                       count(*) filter (
                         where upper(trim(tipo)) in ('CASA','APARTAMENTO','CASA CONDOMINIO')
                       ) as tipo_valido,
                       count(*) filter (
                         where upper(trim(oferta)) in ('VENDA','ALUGUEL','PUBLICADO')
                           and upper(trim(tipo)) in ('CASA','APARTAMENTO','CASA CONDOMINIO')
                           and preco is not null and area_util is not null
                       ) as aproveitado
                from {TABELA}
                where data_coleta >= current_date - {JANELA_DIAS}
                group by 1 order by 2 desc
            """)

    print("\nDiagnóstico concluído (somente leitura).")


if __name__ == "__main__":
    main()
