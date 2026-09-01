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
    [string]$Data = (Get-Date -Format "yyyy-MM-dd")
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
    param([string]$Nome, [string]$Script)

    $pasta = Join-Path $dataDir "coletas\$Nome"
    if (-not (Test-Path $pasta)) { New-Item -ItemType Directory -Force -Path $pasta | Out-Null }
    $saida = Join-Path $pasta "$Data.csv"

    if (Test-Path $saida) {
        Write-Host "[$Nome] $Data.csv já existe — pulando." -ForegroundColor Yellow
        return
    }

    Write-Host "[$Nome] coletando ($Oferta) -> $saida" -ForegroundColor Cyan
    python (Join-Path $raiz $Script) --oferta $Oferta --saida $saida
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$Nome] FALHOU (exit $LASTEXITCODE)" -ForegroundColor Red
        return
    }

    $n = (Get-Content $saida | Measure-Object -Line).Lines - 1
    Write-Host "[$Nome] ok — $n linhas" -ForegroundColor Green
}

if ($Portal -in @("ambos", "df")) { Invoke-Coleta -Nome "df" -Script "scraper\scraperDF.py" }
if ($Portal -in @("ambos", "wi")) { Invoke-Coleta -Nome "wi" -Script "scraper\scraperWI.py" }

Write-Host ""
Write-Host "Coleta de $Data concluída. No início do mês, rode:" -ForegroundColor Cyan
Write-Host "  python fluxo_mensal.py"
