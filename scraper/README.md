# DF Imoveis Scraper

Scraper em Python para capturar anuncios de venda e aluguel no DF Imoveis usando paginacao por URL.

Por padrao ele percorre cidade/bairro nas cidades grandes e cidade direta nas regioes menores para evitar o limite de resultados do site na busca geral.

## Instalar

```powershell
cd "C:\Users\Best Option Notebook\Desktop\61E\Estudos\dfimoveis_scraper"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Rodar

Venda por cidade/bairro, ou cidade direta, ate a primeira pagina vazia de cada escopo:

```powershell
python scraper.py --oferta venda --inicio 1
```

Aluguel por cidade/bairro, ou cidade direta, ate a primeira pagina vazia de cada escopo:

```powershell
python scraper.py --oferta aluguel --inicio 1
```

Venda e aluguel no mesmo CSV, cada escopo ate a primeira pagina vazia:

```powershell
python scraper.py --oferta ambos --inicio 1
```

O mapeamento atual inclui:

- `brasilia`: asa norte, asa sul, altiplano leste, granja do torto, jardim botanico, jardins mangueiral, lago norte, lago sul, noroeste, octogonal, park sul, park way, setor industrial, setor tororo, sig, sudoeste, taquari, vila da telebrasilia, vila planalto, zona civico administrativa, zona rural
- `aguas claras`: ade, areal, arniqueira, norte, sul
- cidades sem bairro: aguas lindas de goias, alphaville, brazlandia, candangolandia, ceilandia, cidade ocidental, cruzeiro, formosa, gama, guara, jardim botanico, luziania, nucleo bandeirante, paranoa, planaltina, planaltina de goias, riacho fundo, samambaia, santa maria, santo antonio do descoberto, sao sebastiao, setor industrial, sobradinho, taguatinga, valparaiso de goias, varjao, vicente pires, vila estrutural

Para rodar apenas uma cidade:

```powershell
python scraper.py --oferta venda --cidades "brasilia"
```

Cidade sem bairro:

```powershell
python scraper.py --oferta venda --cidades "alphaville"
```

Para rodar apenas uma cidade e um bairro:

```powershell
python scraper.py --oferta venda --cidades "brasilia" --bairros "altiplano leste"
```

Se quiser usar a estrategia antiga da busca geral `/df/todos`, informe:

```powershell
python scraper.py --oferta venda --escopo todos
```

Quando `--saida` nao e informado, o arquivo recebe a data da coleta, por exemplo `2026-05-20.csv`.

Para limitar manualmente as paginas por escopo, use `--fim`:

```powershell
python scraper.py --oferta ambos --inicio 1 --fim 5 --saida amostra_paginas.csv
```

## Campos

O CSV segue o layout do arquivo de exemplo:

`id, link, codigo, creci, anunciante, tipo, tipo_imovel, area_util, bairro, cidade, preco, valor_m2, quartos, vagas, latitude, longitude, quadra, data`

Onde `tipo` e a operacao (`venda` ou `aluguel`) e `tipo_imovel` e a categoria do imovel (`casa`, `apartamento`, etc.).

Por padrao o scraper abre a pagina de detalhe de cada imovel, porque codigo, latitude e longitude normalmente ficam no detalhe. Para rodar mais rapido sem detalhe:

```powershell
python scraper.py --oferta venda --inicio 1 --fim 2 --sem-detalhes
```

Tambem da para testar usando os HTMLs salvos:

```powershell
python scraper.py --arquivo-listagem "C:\Users\Best Option Notebook\Downloads\paginacao.html" --saida teste.csv
```

Para validar uma amostra pequena antes de rodar muitas paginas:

```powershell
python scraper.py --oferta venda --inicio 2 --fim 2 --limite 5 --saida amostra.csv
```
