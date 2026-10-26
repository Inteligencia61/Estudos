# Coleta semanal — DF Imóveis e Wimóveis.
#
# Grava em dados/coletas/<portal>/AAAA-MM-DD.csv. O portal vem da PASTA e a
# data do NOME, então nenhum arquivo .py precisa ser editado entre semanas —
# era isso que virava commit para limpar no `git pull`.
#
# Cada arquivo é conferido e carregado no banco logo após a coleta
# (BD/ingestao.py): lote com volume ou qualidade fora do histórico do portal é
# REJEITADO e não entra. O fluxo mensal fica só com estudo e histórico.
#
#   .\coleta_semanal.ps1                 # os dois portais, venda e aluguel
#   .\coleta_semanal.ps1 -Portal df      # só DF Imóveis
#   .\coleta_semanal.ps1 -Oferta venda
#   .\coleta_semanal.ps1 -SemBanco       # só coleta, carga fica para depois

param(
    [ValidateSet("ambos", "df", "wi")] [string]$Portal = "ambos",
    [ValidateSet("ambos", "venda", "aluguel")] [string]$Oferta = "ambos",
    [string]$Data = (Get-Date -Format "yyyy-MM-dd"),
    # O Wimóveis devolve 403 na paginação. Sem --modo-segmentado o scraper lê
    # só a PRIMEIRA página de cada escopo: 56 escopos x 2 ofertas x ~20 cards
    # = ~1.900 anúncios e para. O modo segmentado quebra a busca em
    # categoria/quartos/subtipo/ordenação e deduplica por id.
    [switch]$WiSemSegmentacao,
    # Cada anúncio custa uma requisição a mais para abrir a página de detalhe.
    # Sem detalhe a coleta é MUITO mais rápida, mas vem sem latitude,
    # longitude, código, creci e anunciante — e sem geo o dedupe físico entre
    # portais não funciona. Enriqueça depois com --entrada-csv.
    [switch]$WiComDetalhes,
    # Não carrega no banco (máquina sem credenciais, por exemplo). O arquivo
    # fica em dados/coletas e entra na próxima ingestão.
    [switch]$SemBanco
)

$ErrorActionPreference = "Stop"
$raiz = $PSScriptRoot

# DATA_DIR do .env, se existir; senão ./dados
$dataDir = $env:DATA_DIR
if (-not $dataDir) {
    $envFile = Join-Path $raiz ".env"
    if (Test-Path $envFile) {
        $linha = Select-String -Path $envFile -Pattern '^\s*DATA_DIR\s*=' | Select-Object -First 1
        if ($linha) { $dataDir = ($linha.Line -split '=', 2)[1].Trim().Trim('"') }
    }
}
if (-not $dataDir) { $dataDir = Join-Path $raiz "dados" }

$resultado = @{}

function Invoke-Carga {
    param([string]$Nome, [string]$Arquivo)

    if ($SemBanco) {
        $script:resultado[$Nome] = "coletado, não carregado (-SemBanco)"
        return
    }
    Write-Host "[$Nome] conferindo e carregando no banco..." -ForegroundColor Cyan
    python (Join-Path $raiz "BD\ingestao.py") --arquivo $Arquivo --portal $Nome --oferta $Oferta
    switch ($LASTEXITCODE) {
        0       { $script:resultado[$Nome] = "carregado"
                  Write-Host "[$Nome] carregado" -ForegroundColor Green }
        2       { $script:resultado[$Nome] = "coletado, não carregado (sem credenciais)"
                  Write-Host "[$Nome] sem credenciais do banco — carga fica para depois" -ForegroundColor Yellow }
        3       { $script:resultado[$Nome] = "REJEITADO na conferência"
                  Write-Host "[$Nome] LOTE REJEITADO — veja os [BLOQUEIO] acima" -ForegroundColor Red }
        default { $script:resultado[$Nome] = "ERRO na carga (exit $LASTEXITCODE)"
                  Write-Host "[$Nome] erro na carga (exit $LASTEXITCODE)" -ForegroundColor Red }
    }
}

function Invoke-Coleta {
    param([string]$Nome, [string]$Script, [string[]]$ExtraArgs = @())

    $pasta = Join-Path $dataDir "coletas\$Nome"
    if (-not (Test-Path $pasta)) { New-Item -ItemType Directory -Force -Path $pasta | Out-Null }
    $saida = Join-Path $pasta "$Data.csv"

    if (Test-Path $saida) {
        # A carga é idempotente (hash): rodar de novo completa uma carga que
        # tenha falhado sem refazer a coleta.
        Write-Host "[$Nome] $Data.csv já existe — coleta pulada." -ForegroundColor Yellow
        Invoke-Carga -Nome $Nome -Arquivo $saida
        return
    }

    Write-Host "[$Nome] coletando ($Oferta) $($ExtraArgs -join ' ') -> $saida" -ForegroundColor Cyan
    python (Join-Path $raiz $Script) --oferta $Oferta --saida $saida @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$Nome] FALHOU (exit $LASTEXITCODE)" -ForegroundColor Red
        $script:resultado[$Nome] = "coleta FALHOU"
        return
    }

    $n = (Get-Content $saida | Measure-Object -Line).Lines - 1
    Write-Host "[$Nome] ok — $n linhas" -ForegroundColor Green
    Invoke-Carga -Nome $Nome -Arquivo $saida
}

if ($Portal -in @("ambos", "df")) {
    Invoke-Coleta -Nome "df" -Script "scraper\scraperDF.py"
}

if ($Portal -in @("ambos", "wi")) {
    $wiArgs = @()
    if (-not $WiSemSegmentacao) { $wiArgs += "--modo-segmentado" }
    if (-not $WiComDetalhes)    { $wiArgs += "--sem-detalhes" }
    Invoke-Coleta -Nome "wi" -Script "scraper\scraperWI.py" -ExtraArgs $wiArgs
}

Write-Host ""
Write-Host "Coleta de $Data — resumo:" -ForegroundColor Cyan
foreach ($p in $resultado.Keys) { Write-Host "  $p : $($resultado[$p])" }
Write-Host "No início do mês: python fluxo_mensal.py (estudo + histórico)" -ForegroundColor Cyan
if ($resultado.Values | Where-Object { $_ -match "REJEITADO|FALHOU|ERRO" }) { exit 1 }
