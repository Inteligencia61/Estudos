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

### Chrome normal conectado ao scraper

Com a extensão do transporte nativo instalada, abra o DFImóveis, clique no
ícone da extensão e em **Conectar ao scraper**. Mantenha o painel e a aba do
portal abertos. Execute na raiz de Estudos:

```powershell
python scraper/scraperDF.py --chrome-nativo --oferta venda --cidades brasilia --bairros "asa sul" --fim 2 --limite 31 --delay 5 --saida amostra_df.csv
```

O Python continua controlando os filtros, a paginação, os detalhes de cada
anúncio e a gravação do CSV. A extensão entrega o HTML da aba normal.
Código e coordenadas são preenchidos quando disponíveis no anúncio.
Falhas de conexão, bloqueios e páginas sem a estrutura esperada interrompem
a coleta. O progresso lido fica em um arquivo .part, separado do CSV final.

### Retomada automática

Nas coletas online, o scraper grava os anúncios concluídos e a página em
processamento em um arquivo ao lado da saída, por exemplo
dfimoveis_completo.csv.checkpoint.sqlite3.

Se o terminal fechar, o computador reiniciar ou a coleta for interrompida,
conecte novamente o Chrome e repita o mesmo comando com o mesmo --saida.
Os filtros, limites e a opção de detalhes devem ser os mesmos. Pausas,
timeout e tentativas podem ser ajustados na retomada.

O arquivo de checkpoint é necessário para retomar. Mantenha-o junto da
saída. Um checkpoint concluído permite reexportar o CSV sem consultar o
portal. Para começar uma nova coleta, use outro nome de saída.
Arquivos .part antigos, gerados antes desse recurso, não contêm esse estado.

Para a coleta semanal DFImóveis com esse transporte:

```powershell
.\coleta_semanal.ps1 -Portal df -DfChromeNativo -SemBanco
```

O exemplo usa -SemBanco para apenas gerar a coleta. A execução com carga
mantém a conferência de qualidade existente no pipeline.

Se a coleta HTTP receber o desafio da Cloudflare, há um modo de teste com
Google Chrome visível. A partir da raiz de Estudos, com o Chrome instalado:

```powershell
python -m pip install playwright
python scraper/scraperDF.py --chrome --oferta venda --cidades brasilia --bairros "asa sul" --fim 2 --limite 31 --sem-detalhes --timeout 90 --saida amostra_df.csv
```

Se o portal pedir verificação, conclua-a na janela do Chrome durante o tempo
de `--timeout`. Uma coleta interrompida mantém os anúncios já lidos em um
arquivo `.part`. O CSV final é publicado após a conclusão da coleta solicitada.

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
