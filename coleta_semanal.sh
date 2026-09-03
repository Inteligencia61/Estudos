#!/usr/bin/env bash
#
# Coleta semanal — DF Imóveis e Wimóveis. Versão macOS/Linux.
#
# Equivalente ao coleta_semanal.ps1 (Windows). Escrito para o bash 3.2 que a
# Apple ainda distribui em Macs antigos: sem arrays associativos, sem
# ${var,,}, sem readlink -f.
#
# Grava em dados/coletas/<portal>/AAAA-MM-DD.csv. O portal vem da PASTA e a
# data do NOME, então nenhum arquivo .py precisa ser editado entre semanas.
#
#   chmod +x coleta_semanal.sh      (uma vez só)
#
#   ./coleta_semanal.sh                    # os dois portais, venda e aluguel
#   ./coleta_semanal.sh --portal df        # só DF Imóveis
#   ./coleta_semanal.sh --oferta venda
#   ./coleta_semanal.sh --setup            # cria o .venv e instala as libs
#
# Para abrir com dois cliques no Finder, renomeie para coleta_semanal.command.

set -eu

# ------------------------------------------------------------------
# Padrões
# ------------------------------------------------------------------
PORTAL="ambos"
OFERTA="ambos"
DATA="$(date +%Y-%m-%d)"
WI_SEM_SEGMENTACAO=0
WI_COM_DETALHES=0
SETUP=0

uso() {
    cat <<'FIM'
Uso: ./coleta_semanal.sh [opções]

  --portal df|wi|ambos     padrão: ambos
  --oferta venda|aluguel|ambos
  --data AAAA-MM-DD        padrão: hoje
  --wi-com-detalhes        abre a página de cada anúncio do Wimóveis.
                           Traz latitude, longitude, código, creci e
                           anunciante — e é MUITO mais demorado (uma
                           requisição por anúncio).
  --wi-sem-segmentacao     desliga o --modo-segmentado do Wimóveis.
                           NÃO recomendado: o site devolve 403 na paginação,
                           e sem segmentação a coleta lê só a primeira página
                           de cada escopo e para em ~1.900 anúncios.
  --setup                  cria o .venv e instala requirements
  -h, --help               esta ajuda
FIM
}

while [ $# -gt 0 ]; do
    case "$1" in
        --portal) PORTAL="$2"; shift 2 ;;
        --oferta) OFERTA="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --wi-com-detalhes) WI_COM_DETALHES=1; shift ;;
        --wi-sem-segmentacao) WI_SEM_SEGMENTACAO=1; shift ;;
        --setup) SETUP=1; shift ;;
        -h|--help) uso; exit 0 ;;
        *) echo "Opção desconhecida: $1" >&2; uso; exit 2 ;;
    esac
done

case "$PORTAL" in df|wi|ambos) ;; *) echo "portal inválido: $PORTAL" >&2; exit 2 ;; esac
case "$OFERTA" in venda|aluguel|ambos) ;; *) echo "oferta inválida: $OFERTA" >&2; exit 2 ;; esac

# ------------------------------------------------------------------
# Onde estamos (macOS não tem `readlink -f`)
# ------------------------------------------------------------------
RAIZ="$(cd "$(dirname "$0")" && pwd)"
cd "$RAIZ"

# ------------------------------------------------------------------
# Cores só quando a saída é um terminal
# ------------------------------------------------------------------
if [ -t 1 ]; then
    C_INFO=$'\033[36m'; C_OK=$'\033[32m'; C_AVISO=$'\033[33m'; C_ERRO=$'\033[31m'; C_OFF=$'\033[0m'
else
    C_INFO=''; C_OK=''; C_AVISO=''; C_ERRO=''; C_OFF=''
fi

msg()   { printf '%s%s%s\n' "$C_INFO" "$1" "$C_OFF"; }
ok()    { printf '%s%s%s\n' "$C_OK" "$1" "$C_OFF"; }
aviso() { printf '%s%s%s\n' "$C_AVISO" "$1" "$C_OFF"; }
erro()  { printf '%s%s%s\n' "$C_ERRO" "$1" "$C_OFF" >&2; }

# ------------------------------------------------------------------
# Python: o .venv do projeto tem prioridade
# ------------------------------------------------------------------
PY=""
if [ -x "$RAIZ/.venv/bin/python" ]; then
    PY="$RAIZ/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    erro "python3 não encontrado."
    erro "Instale pelo Homebrew (brew install python) ou pelo instalador em python.org."
    exit 1
fi

# Os scrapers usam anotações modernas, mas têm 'from __future__ import
# annotations', então 3.9 basta. Abaixo disso não roda.
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    erro "Python muito antigo: $("$PY" --version 2>&1). É preciso 3.9 ou superior."
    erro "Em Mac antigo, o caminho mais simples é o instalador oficial do python.org,"
    erro "que ainda publica builds para versões antigas do macOS."
    exit 1
fi

if [ "$SETUP" -eq 1 ]; then
    msg "Criando .venv e instalando dependências..."
    "$PY" -m venv "$RAIZ/.venv"
    "$RAIZ/.venv/bin/python" -m pip install --upgrade pip
    "$RAIZ/.venv/bin/python" -m pip install -r "$RAIZ/scraper/requirements.txt"
    ok "Pronto. Rode ./coleta_semanal.sh normalmente."
    exit 0
fi

# Aviso antecipado: erro de import no meio da coleta desperdiça a rodada.
if ! "$PY" -c 'import requests, bs4' >/dev/null 2>&1; then
    erro "Faltam dependências (requests / beautifulsoup4)."
    erro "Rode: ./coleta_semanal.sh --setup"
    exit 1
fi

# ------------------------------------------------------------------
# Onde gravar: DATA_DIR do ambiente, do .env, ou ./dados
# ------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-}"
if [ -z "$DATA_DIR" ] && [ -f "$RAIZ/.env" ]; then
    DATA_DIR="$(grep -E '^[[:space:]]*DATA_DIR[[:space:]]*=' "$RAIZ/.env" 2>/dev/null \
                | head -1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')"
fi
[ -n "$DATA_DIR" ] || DATA_DIR="$RAIZ/dados"

# ------------------------------------------------------------------
# Coleta de um portal
# ------------------------------------------------------------------
coletar() {
    nome="$1"; script="$2"; shift 2

    pasta="$DATA_DIR/coletas/$nome"
    mkdir -p "$pasta"
    saida="$pasta/$DATA.csv"

    if [ -f "$saida" ]; then
        aviso "[$nome] $DATA.csv já existe — pulando."
        return 0
    fi

    msg "[$nome] coletando ($OFERTA) $* -> $saida"

    # `set -e` mataria o script inteiro se um portal falhasse; aqui o outro
    # portal ainda tem chance de rodar.
    if "$PY" "$RAIZ/$script" --oferta "$OFERTA" --saida "$saida" "$@"; then
        :
    else
        erro "[$nome] FALHOU (exit $?)"
        return 0
    fi

    if [ -f "$saida" ]; then
        total=$(( $(wc -l < "$saida") - 1 ))
        ok "[$nome] ok — $total linhas"
        if [ "$total" -lt 100 ]; then
            aviso "[$nome] volume baixo. Rodada parcial ou bloqueio do portal?"
        fi
    else
        erro "[$nome] terminou sem gerar arquivo."
    fi
}

# ------------------------------------------------------------------
# Execução
# ------------------------------------------------------------------
if [ "$PORTAL" = "ambos" ] || [ "$PORTAL" = "df" ]; then
    coletar "df" "scraper/scraperDF.py"
fi

if [ "$PORTAL" = "ambos" ] || [ "$PORTAL" = "wi" ]; then
    # Sem arrays associativos (bash 3.2): monta os argumentos em variáveis.
    WI_ARGS=""
    [ "$WI_SEM_SEGMENTACAO" -eq 1 ] || WI_ARGS="--modo-segmentado"
    if [ "$WI_COM_DETALHES" -eq 0 ]; then
        WI_ARGS="$WI_ARGS --sem-detalhes"
    fi
    # shellcheck disable=SC2086
    coletar "wi" "scraper/scraperWI.py" $WI_ARGS
fi

echo
msg "Coleta de $DATA concluída."
msg "Os CSVs estão em: $DATA_DIR/coletas/"
msg "No início do mês, na máquina do banco: python fluxo_mensal.py"
