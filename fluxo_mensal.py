# -*- coding: utf-8 -*-
"""
Fluxo mensal completo — um comando, nenhuma edição de código.

    semana:  coleta_semanal.ps1   -> dados/coletas/<portal>/AAAA-MM-DD.csv
    mês:     python fluxo_mensal.py

Etapas, na ordem:

    1. ingestão   arquivos pendentes -> tabela imoveis (idempotente)
    2. estudo     analytics.estudo_metricas      (preço pedido por segmento)
    3. histórico  analytics.listagem_historico   (vida de cada anúncio)
                  analytics.mercado_metricas     (DOM, absorção, desconto)
    4. conferência   diagnóstico somente-leitura

Cada etapa pode ser pulada. Rodar duas vezes é seguro: a ingestão pula
arquivo já carregado (hash) e as métricas são reescritas por escopo.

    python fluxo_mensal.py --dry-run       # mostra o que faria
    python fluxo_mensal.py --so-estudo     # pula ingestão
    python fluxo_mensal.py --meses-historico 24
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "BD"))
sys.path.insert(0, str(RAIZ / "analise" / "acionador"))


def cabecalho(n: int, titulo: str) -> float:
    print("\n" + "=" * 72)
    print(f"ETAPA {n} — {titulo}")
    print("=" * 72)
    return time.time()


def fim(t0: float) -> None:
    print(f"  ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fluxo mensal do estudo de mercado")
    ap.add_argument("--dry-run", action="store_true",
                    help="ingestão em modo simulação e nenhuma gravação de métrica")
    ap.add_argument("--pular-ingestao", action="store_true")
    ap.add_argument("--pular-estudo", action="store_true")
    ap.add_argument("--pular-historico", action="store_true")
    ap.add_argument("--pular-diagnostico", action="store_true")
    ap.add_argument("--so-estudo", action="store_true",
                    help="atalho: só a etapa 2")
    ap.add_argument("--meses-historico", type=int, default=12)
    ap.add_argument("--pasta", type=Path, help="raiz das coletas")
    args = ap.parse_args()

    if args.so_estudo:
        args.pular_ingestao = args.pular_historico = args.pular_diagnostico = True

    falhas = []

    # ---------------------------------------------------------- 1. ingestão
    if not args.pular_ingestao:
        t0 = cabecalho(1, "INGESTÃO das coletas")
        try:
            import ingestao
            cargas = ingestao.descobrir(args.pasta)
            if not cargas:
                print("  nenhum arquivo pendente encontrado.")
            else:
                print(f"  {len(cargas)} arquivo(s) encontrado(s):")
                for c in cargas:
                    print(f"    {c['portal']:3} {c['data_coleta']}  {c['arquivo'].name}")
                ingestao.processar(cargas, dry_run=args.dry_run)
        except Exception:
            traceback.print_exc()
            falhas.append("ingestão")
        fim(t0)

    # ---------------------------------------------------------- 2. estudo
    if not args.pular_estudo:
        t0 = cabecalho(2, "ESTUDO de preço pedido (acionador)")
        try:
            from acionador import EstudoMercado
            em = EstudoMercado()
            print(f"  meses-alvo: {em.meses_alvo}")
            if args.dry_run:
                print("  [dry-run] pipeline não executado")
            else:
                em.enviar_banco()
        except Exception:
            traceback.print_exc()
            falhas.append("estudo")
        fim(t0)

    # ---------------------------------------------------------- 3. histórico
    if not args.pular_historico:
        t0 = cabecalho(3, "HISTÓRICO de listagem e métricas de mercado")
        try:
            from listagem_historico import HistoricoListagens
            if args.dry_run:
                print("  [dry-run] tabelas não reconstruídas")
            else:
                HistoricoListagens(meses=args.meses_historico).construir()
        except Exception:
            traceback.print_exc()
            falhas.append("histórico")
        fim(t0)

    # ---------------------------------------------------------- 4. conferência
    if not args.pular_diagnostico:
        t0 = cabecalho(4, "CONFERÊNCIA (somente leitura)")
        try:
            import diagnostico_portais
            diagnostico_portais.main()
        except Exception:
            traceback.print_exc()
            falhas.append("diagnóstico")
        fim(t0)

    print("\n" + "=" * 72)
    if falhas:
        print("CONCLUÍDO COM FALHAS em: " + ", ".join(falhas))
        sys.exit(1)
    print("FLUXO MENSAL CONCLUÍDO.")


if __name__ == "__main__":
    main()
