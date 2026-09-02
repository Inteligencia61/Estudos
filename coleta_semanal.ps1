# Coleta semanal — DF Imóveis e Wimóveis.
#
# Grava em dados/coletas/<portal>/AAAA-MM-DD.csv. O portal vem da PASTA e a
# data do NOME, então nenhum arquivo .py precisa ser editado entre semanas —
# era isso que virava commit para limpar no `git pull`.
#
#   .\coleta_semanal.ps1                 # os dois portais, venda e aluguel
#   .\coleta_semanal.ps1 -Portal df      # só DF Imóveis
#   .\coleta_semanal.ps1 -Oferta venda

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
    [switch]$WiComDetalhes
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

function Invoke-Coleta {
    param([string]$Nome, [string]$Script, [string[]]$ExtraArgs = @())

    $pasta = Join-Path $dataDir "coletas\$Nome"
    if (-not (Test-Path $pasta)) { New-Item -ItemType Directory -Force -Path $pasta | Out-Null }
    $saida = Join-Path $pasta "$Data.csv"

    if (Test-Path $saida) {
        Write-Host "[$Nome] $Data.csv já existe — pulando." -ForegroundColor Yellow
        return
    }

    Write-Host "[$Nome] coletando ($Oferta) $($ExtraArgs -join ' ') -> $saida" -ForegroundColor Cyan
    python (Join-Path $raiz $Script) --oferta $Oferta --saida $saida @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$Nome] FALHOU (exit $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    $n = (Get-Content $saida | Measure-Object -Line).Lines - 1
    Write-Host "[$Nome] ok — $n linhas" -ForegroundColor Green
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
Write-Host "Coleta de $Data concluída. No início do mês, rode:" -ForegroundColor Cyan
Write-Host "  python fluxo_mensal.py"
