# -*- coding: utf-8 -*-
"""
Testes da ingestão sem banco: conferência do lote, código corrompido e
descoberta de arquivos.

    python BD/testes_ingestao.py

O histórico sintético reproduz o perfil real medido no banco em out/2026
(DF: ~38 mil venda, ~7,5 mil aluguel, 12% sem latitude; WI: 39% sem bairro,
100% sem latitude) e os incidentes reais que a conferência precisa barrar.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingestao  # noqa: E402


def _historico(venda: int, aluguel: int, pct_sem_bairro: float = 0.0,
               pct_sem_lat: float = 12.4, n: int = 6) -> pd.DataFrame:
    linhas = []
    for i in range(n):
        d = date(2026, 9, 1) + timedelta(days=3 * i)
        for oferta, q in (("VENDA", venda), ("ALUGUEL", aluguel)):
            linhas.append(dict(data_coleta=d, oferta=oferta, linhas=q, sem_tipo=0,
                               sem_bairro=int(q * pct_sem_bairro / 100),
                               sem_lat=int(q * pct_sem_lat / 100)))
    return pd.DataFrame(linhas)


def _perfil(venda: int, aluguel: int, sem_tipo=0.0, sem_bairro=0.0, sem_lat=12.4) -> dict:
    return {"linhas": {"VENDA": venda, "ALUGUEL": aluguel},
            "sem_tipo": sem_tipo, "sem_bairro": sem_bairro, "sem_lat": sem_lat}


AMBAS = ("VENDA", "ALUGUEL")
HIST_DF = _historico(38200, 7500)
HIST_WI = _historico(12600, 4720, pct_sem_bairro=38.9, pct_sem_lat=100.0)


def teste_semana_normal_passa():
    b, a = ingestao.conferir_lote(_perfil(38900, 7650), 46600, HIST_DF, AMBAS)
    assert not b and not a, (b, a)
    # WI com 39% sem bairro é o normal dele, não bloqueio.
    b, a = ingestao.conferir_lote(_perfil(12650, 4700, sem_bairro=39.0, sem_lat=100.0),
                                  17400, HIST_WI, AMBAS)
    assert not b and not a, (b, a)
    print("OK  semana normal passa (DF e WI)")


def teste_coleta_parcial_bloqueia():
    # DF 18/08: 573 linhas, só venda.
    b, _ = ingestao.conferir_lote(_perfil(573, 0), 575, HIST_DF, AMBAS)
    assert any("VENDA" in x for x in b) and any("ALUGUEL" in x for x in b), b
    # DF 08/09: aluguel com 1/3 do volume.
    b, _ = ingestao.conferir_lote(_perfil(37525, 2337), 39900, HIST_DF, AMBAS)
    assert len(b) == 1 and "ALUGUEL" in b[0], b
    # DF 30/09: venda -13% é aviso, não bloqueio.
    b, a = ingestao.conferir_lote(_perfil(33402, 7803), 41300, HIST_DF, AMBAS)
    assert not b and len(a) == 1 and "VENDA" in a[0], (b, a)
    print("OK  coleta parcial bloqueia; queda moderada só avisa")


def teste_coleta_so_de_venda():
    # -Oferta venda: aluguel ausente não é problema.
    b, _ = ingestao.conferir_lote(_perfil(38000, 0), 38050, HIST_DF, ("VENDA",))
    assert not b, b
    print("OK  coleta só de venda não cobra aluguel")


def teste_qualidade_bloqueia():
    # Layout trocado: tipo vazio.
    b, _ = ingestao.conferir_lote(_perfil(38000, 7500, sem_tipo=60.0), 45600, HIST_DF, AMBAS)
    assert any("sem_tipo" in x for x in b), b
    # DF perdendo bairro.
    b, _ = ingestao.conferir_lote(_perfil(38000, 7500, sem_bairro=25.0), 45600, HIST_DF, AMBAS)
    assert any("sem_bairro" in x for x in b), b
    # Latitude sumindo é aviso.
    b, a = ingestao.conferir_lote(_perfil(38000, 7500, sem_lat=80.0), 45600, HIST_DF, AMBAS)
    assert not b and any("sem_lat" in x for x in a), (b, a)
    # Arquivo de 2,8 mil linhas com 0 aproveitadas (relatório _segmentos).
    b, _ = ingestao.conferir_lote({"linhas": {}, "sem_tipo": 0, "sem_bairro": 0,
                                   "sem_lat": 0}, 2800, HIST_WI, AMBAS)
    assert any("aproveitamento" in x for x in b), b
    print("OK  qualidade e aproveitamento bloqueiam")


def teste_sem_historico():
    vazio = pd.DataFrame()
    b, _ = ingestao.conferir_lote(_perfil(500, 100), 600, vazio, AMBAS)
    assert not b, b   # primeira carga do portal: sem base de volume
    b, _ = ingestao.conferir_lote(_perfil(500, 100, sem_tipo=30.0), 600, vazio, AMBAS)
    assert any("sem_tipo" in x for x in b), b
    print("OK  sem histórico vale só o limite absoluto")


def teste_codigo_corrompido():
    bruto = pd.DataFrame({
        "id": ["111111", "222222", "333333", "444444"],
        "codigo": ["AP0123", "Piso Frio", "Churrasqueira, Piscina", "X" * 40 + "1"],
        "link": ["https://x/111111"] * 4,
        "tipo": ["venda"] * 4, "tipo_imovel": ["apartamento"] * 4,
        "preco": ["500000"] * 4, "area_util": ["80"] * 4, "bairro": ["GUARA"] * 4,
    })
    out = ingestao.normalizar(bruto, "df", "2026-10-26")
    assert out["codigo"].tolist() == ["AP0123", "222222", "333333", "444444"], out["codigo"].tolist()
    print("OK  código com texto de descrição cai para o id do anúncio")


def teste_descobrir_ignora_segmentos():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "wi").mkdir()
        for nome in ("2026-10-26.csv", "2026-10-26_segmentos.csv", "copia 2026-10-26.csv"):
            (base / "wi" / nome).write_text("a\n1\n", encoding="utf-8")
        achados = ingestao.descobrir(base)
    assert [a["arquivo"].name for a in achados] == ["2026-10-26.csv"], achados
    assert achados[0]["data_coleta"] == "2026-10-26"
    print("OK  descobrir aceita só AAAA-MM-DD.csv")


if __name__ == "__main__":
    teste_semana_normal_passa()
    teste_coleta_parcial_bloqueia()
    teste_coleta_so_de_venda()
    teste_qualidade_bloqueia()
    teste_sem_historico()
    teste_codigo_corrompido()
    teste_descobrir_ignora_segmentos()
    print("\nTODOS OS TESTES PASSARAM")
