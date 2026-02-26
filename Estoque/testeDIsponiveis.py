# check_disponibilidade_imoveis.py
# ============================================================
# Testa uma lista de códigos e diz quais estão disponíveis
# no endpoint de "imóveis vagos/disponíveis" do Imoview.
#
# Saídas:
# - resultado_disponibilidade.csv
# - disponiveis.txt
# - indisponiveis.txt
#
# Requisitos:
#   pip install requests
# ============================================================

from __future__ import annotations

import csv
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests


# =========================
# CONFIG (AJUSTE AQUI)
# =========================
ENDPOINT_URL = "https://api.imoview.com.br/Imovel/RetornarImoveisDisponiveis"

# Headers do Imoview
# - 'chave' é obrigatório
# - Algumas rotas App_ exigem 'codigoacesso'
HEADERS = {
    "chave": "a4ff7c378eff87533b123d25c9b6f088",
    # "codigoacesso": "SEU_CODIGO_ACESSO_AQUI",  # descomente se o endpoint exigir
    "Content-Type": "application/json",
}

# Parâmetros obrigatórios do endpoint (conforme sua documentação)
FINALIDADE = 2          # 1 = ALUGUEL | 2 = VENDA
NUM_REGISTROS = 20      # máximo 20 (conforme docs)
SITUACAO = 1            # se seu endpoint suportar (1 = Vago/Disponível). Se não existir, deixe None.

# Controle de paginação e robustez
TIMEOUT = 45
RETRIES = 4
SLEEP_ENTRE_REQS = 0.25
BACKOFF_BASE = 1.6

# Para evitar URL grande demais, enviamos os códigos em lotes.
# (A rota aceita "codigosimoveis" separados por vírgula.)
BATCH_SIZE_CODIGOS = 120


# =========================
# SUA LISTA DE CÓDIGOS
# (mantive exatamente os que você mandou; o script também deduplica)
# =========================
CODIGOS_RAW = """
11374
11781
11732
11716
11654
11143
10654
11163
10438
10439
10229
11685
11663
11391
11666
11406
10075
10585
11263
11223
11543
11479
11455
11597
11330
10725
11231
10993
5292
9111
10348
9567
9242
8695
6140
11730
10702
11715
11677
9892
10199
11635
11299
10108
11144
11618
5447
11381
11631
11341
11683
11647
11674
11387
11301
11602
3872
9241
10698
10375
11541
11425
11316
10784
11336
11254
9606
11052
10786
10247
5235
11153
11109
11110
10920
9624
""".strip()


# =========================
# Utils
# =========================
def parse_codigos(texto: str) -> List[int]:
    cods: List[int] = []
    for line in texto.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            cods.append(int(s))
        except ValueError:
            pass
    return cods


def chunks(lst: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def request_with_retry(
    session: requests.Session,
    url: str,
    json_payload: Dict,
    headers: Dict,
    timeout: int,
    retries: int,
    backoff_base: float,
) -> Dict:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = session.post(url, json=json_payload, headers=headers, timeout=timeout)
            # Erros HTTP
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            if not isinstance(data, dict):
                raise RuntimeError("Resposta não é JSON objeto (dict).")
            return data
        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_s = (backoff_base ** attempt)
                time.sleep(sleep_s)
                continue
            break
    raise RuntimeError(f"Falha após {retries+1} tentativas: {last_err}") from last_err


def extrair_codigos_disponiveis(resp: Dict) -> Set[int]:
    """
    A resposta tem o formato:
    {
      "quantidade": int,
      "lista": [ { "codigo": int, ... }, ... ]
    }
    """
    out: Set[int] = set()
    lista = resp.get("lista", [])
    if isinstance(lista, list):
        for item in lista:
            if isinstance(item, dict) and "codigo" in item:
                try:
                    out.add(int(item["codigo"]))
                except Exception:
                    pass
    return out


def montar_payload(codigos: List[int], pagina: int) -> Dict:
    payload = {
        "finalidade": FINALIDADE,
        "numeropagina": pagina,
        "numeroregistros": NUM_REGISTROS,
        "codigosimoveis": ",".join(str(c) for c in codigos),
    }
    # Se o seu endpoint tiver o parâmetro "situacao" conforme a doc:
    if SITUACAO is not None:
        payload["situacao"] = SITUACAO
    return payload


def buscar_disponiveis_para_lote(
    session: requests.Session,
    codigos_lote: List[int],
) -> Set[int]:
    """
    Faz paginação até não vir mais itens.
    Retorna o conjunto de códigos disponíveis encontrados para esse lote.
    """
    encontrados: Set[int] = set()
    pagina = 1

    while True:
        payload = montar_payload(codigos_lote, pagina)
        resp = request_with_retry(
            session=session,
            url=ENDPOINT_URL,
            json_payload=payload,
            headers=HEADERS,
            timeout=TIMEOUT,
            retries=RETRIES,
            backoff_base=BACKOFF_BASE,
        )

        cods = extrair_codigos_disponiveis(resp)
        if not cods:
            break

        encontrados |= cods

        # Se veio menos que o limite, provavelmente acabou
        lista = resp.get("lista", [])
        if not isinstance(lista, list) or len(lista) < NUM_REGISTROS:
            break

        pagina += 1
        time.sleep(SLEEP_ENTRE_REQS)

    return encontrados


def main() -> None:
    if "COLE_AQUI_A_URL_DO_ENDPOINT" in ENDPOINT_URL:
        raise SystemExit(
            "ERRO: Você precisa configurar ENDPOINT_URL com a URL real do endpoint."
        )
    if HEADERS.get("chave", "").startswith("SUA_CHAVE"):
        raise SystemExit(
            "ERRO: Você precisa configurar HEADERS['chave'] com sua chave do Imoview."
        )

    codigos = parse_codigos(CODIGOS_RAW)
    codigos_unicos = sorted(set(codigos))

    print(f"Total informados: {len(codigos)}")
    print(f"Total únicos: {len(codigos_unicos)}")
    print(f"Lotes de {BATCH_SIZE_CODIGOS}...")

    disponiveis: Set[int] = set()

    with requests.Session() as session:
        for idx, lote in enumerate(chunks(codigos_unicos, BATCH_SIZE_CODIGOS), start=1):
            print(f"  - Lote {idx}: {len(lote)} códigos")
            achados = buscar_disponiveis_para_lote(session, lote)
            disponiveis |= achados
            time.sleep(SLEEP_ENTRE_REQS)

    indisponiveis = [c for c in codigos_unicos if c not in disponiveis]

    print("\n==================== RESULTADO ====================")
    print(f"Disponíveis:   {len(disponiveis)}")
    print(f"Indisponíveis: {len(indisponiveis)}")

    # CSV
    with open("resultado_disponibilidade.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["codigo", "disponivel"])
        for c in codigos_unicos:
            w.writerow([c, "SIM" if c in disponiveis else "NAO"])

    # TXT
    with open("disponiveis.txt", "w", encoding="utf-8") as f:
        for c in sorted(disponiveis):
            f.write(f"{c}\n")

    with open("indisponiveis.txt", "w", encoding="utf-8") as f:
        for c in indisponiveis:
            f.write(f"{c}\n")

    print("\nArquivos gerados:")
    print(" - resultado_disponibilidade.csv")
    print(" - disponiveis.txt")
    print(" - indisponiveis.txt")


if __name__ == "__main__":
    main()
