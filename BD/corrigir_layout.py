# -*- coding: utf-8 -*-
"""
Reparo do desalinhamento `oferta` / `tipo` já gravado na tabela `imoveis`.

CONTEXTO (medido no banco em 2026-09-01)

Os scrapers passaram a emitir `tipo` = venda/aluguel e `tipo_imovel` =
apartamento/casa, mas a carga continuou tratando `tipo` como tipo do imóvel.
O corte é limpo:

    até 2026-08-13   layout antigo, oferta preenchida — base saudável
    de 2026-08-18    layout novo com mapeamento antigo — 162.413 linhas
                     com `oferta` vazia e `tipo` contendo VENDA/ALUGUEL

Duas classes de estrago, com reparos diferentes:

  CLASSE A — colunas trocadas entre si  (~45,6 mil linhas)
      oferta ∈ {APARTAMENTO, CASA, ...} e tipo ∈ {VENDA, ALUGUEL, ...}
      Reparo: trocar os dois campos. Nada se perde.

  CLASSE B — tipo do imóvel PERDIDO NA CARGA  (~162,4 mil linhas)
      oferta vazia e tipo = VENDA/ALUGUEL/LANCAMENTO.
      O `tipo_imovel` do CSV nunca chegou ao banco: não existe informação
      no banco para reconstruí-lo.

      Reparo correto: recarregar os CSVs originais dessas datas com
      `BD/ingestao.py`, que já entende o layout novo. `--recuperar-oferta`
      é o paliativo: salva a oferta e marca o tipo como desconhecido,
      devolvendo as linhas ao histórico de DOM/absorção (que não filtra
      por tipo) sem fingir um tipo que não existe.

Nada é escrito sem `--aplicar`. O padrão é relatório.

    python BD/corrigir_layout.py                     # só relatório
    python BD/corrigir_layout.py --swap --aplicar    # classe A
    python BD/corrigir_layout.py --recuperar-oferta --aplicar   # classe B
"""
from __future__ import annotations

import argparse
from contextlib import closing

from ingestao import MAPA_OFERTA, TABELA, VOCAB_TIPO, conectar

LISTA_TIPO = sorted(VOCAB_TIPO)
LISTA_OFERTA = sorted(MAPA_OFERTA.keys())


def relatorio(cur) -> None:
    print("=" * 72)
    print("ESTADO DE `oferta` / `tipo`")
    print("=" * 72)

    cur.execute(f"""
        select
          count(*) as total,
          count(*) filter (
            where upper(trim(oferta)) = any(%s) and upper(trim(tipo)) = any(%s)
          ) as classe_a_trocado,
          count(*) filter (
            where coalesce(trim(oferta),'') = '' and upper(trim(tipo)) = any(%s)
          ) as classe_b_oferta_perdida,
          count(*) filter (
            where upper(trim(oferta)) = any(%s) and upper(trim(tipo)) = any(%s)
          ) as saudavel
        from {TABELA}
    """, (LISTA_TIPO, LISTA_OFERTA, LISTA_OFERTA, LISTA_OFERTA, LISTA_TIPO))
    total, a, b, ok = cur.fetchone()
    print(f"  total de linhas          {total:>10,}")
    print(f"  saudáveis                {ok:>10,}  ({ok/total:.1%})")
    print(f"  CLASSE A (trocadas)      {a:>10,}  -> --swap resolve")
    print(f"  CLASSE B (oferta vazia)  {b:>10,}  -> recarregar CSV; "
          f"--recuperar-oferta é paliativo")

    if b:
        print("\n  Datas afetadas pela CLASSE B (recarregue estes CSVs):")
        cur.execute(f"""
            select data_coleta, count(*) as linhas
            from {TABELA}
            where coalesce(trim(oferta),'') = '' and upper(trim(tipo)) = any(%s)
            group by 1 order by 1
        """, (LISTA_OFERTA,))
        for dt, n in cur.fetchall():
            print(f"    {dt}   {n:>8,} linhas")


def aplicar_swap(cur) -> int:
    """CLASSE A: troca oferta <-> tipo onde os dois estão claramente invertidos."""
    cur.execute(f"""
        update {TABELA}
           set oferta = upper(trim(tipo)),
               tipo   = upper(trim(oferta))
         where upper(trim(oferta)) = any(%s)
           and upper(trim(tipo))   = any(%s)
    """, (LISTA_TIPO, LISTA_OFERTA))
    return cur.rowcount


def aplicar_recuperacao(cur) -> int:
    """
    CLASSE B: move a oferta de `tipo` para `oferta` e zera `tipo`.

    Deixar `tipo` com "VENDA" é pior do que deixá-lo nulo: o acionador filtra
    `tipo in ('CASA','APARTAMENTO','CASA CONDOMINIO')`, então a linha fica
    invisível de qualquer jeito — mas o histórico de DOM/absorção, que não
    filtra tipo, volta a enxergá-la com a oferta certa.
    """
    cur.execute(f"""
        update {TABELA}
           set oferta = case upper(trim(tipo))
                          when 'LANCAMENTO' then 'VENDA'
                          when 'LANÇAMENTO' then 'VENDA'
                          else upper(trim(tipo))
                        end,
               tipo   = null
         where coalesce(trim(oferta),'') = ''
           and upper(trim(tipo)) = any(%s)
    """, (LISTA_OFERTA,))
    return cur.rowcount


def main() -> None:
    ap = argparse.ArgumentParser(description="Reparo de oferta/tipo em `imoveis`")
    ap.add_argument("--swap", action="store_true", help="CLASSE A: desfaz a troca")
    ap.add_argument("--recuperar-oferta", action="store_true",
                    help="CLASSE B: salva a oferta, marca tipo como nulo")
    ap.add_argument("--aplicar", action="store_true",
                    help="grava de fato (sem isto, apenas relatório)")
    args = ap.parse_args()

    with closing(conectar()) as conn:
        with conn.cursor() as cur:
            cur.execute("set statement_timeout = '600s'")
            relatorio(cur)

            if not (args.swap or args.recuperar_oferta):
                print("\nNada solicitado. Use --swap e/ou --recuperar-oferta.")
                return
            if not args.aplicar:
                print("\n[DRY-RUN] nada gravado. Repita com --aplicar.")
                return

            total = 0
            if args.swap:
                n = aplicar_swap(cur)
                print(f"\n[OK] CLASSE A: {n:,} linhas corrigidas.")
                total += n
            if args.recuperar_oferta:
                n = aplicar_recuperacao(cur)
                print(f"[OK] CLASSE B: {n:,} linhas com oferta recuperada "
                      f"(tipo marcado como nulo).")
                total += n

            conn.commit()
            print(f"\nCommit feito. {total:,} linhas alteradas.")
            print("Rode o diagnóstico para conferir:")
            print("  python analise/acionador/diagnostico_portais.py")


if __name__ == "__main__":
    main()
