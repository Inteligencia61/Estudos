# Acionador — Estudo de Mercado

Pipeline de estudo mercadológico da 61: lê o painel de anúncios coletados
(`imoveis`), trata, segmenta e grava métricas em `analytics`.

```
coleta_semanal.ps1            ->  dados/coletas/<portal>/AAAA-MM-DD.csv
  scraper/scraperDF.py                (portal df)
  scraper/scraperWI.py                (portal wi)

fluxo_mensal.py               ->  roda as três etapas abaixo, em ordem
  BD/ingestao.py              ->  tabela imoveis (painel: anúncio x semana)
  analise/acionador/
    acionador.py              ->  analytics.estudo_metricas     (preço pedido por segmento)
    listagem_historico.py     ->  analytics.listagem_historico   (vida de cada anúncio)
                                  analytics.mercado_metricas     (DOM, absorção, desconto)
    diagnostico_portais.py    ->  conferência somente-leitura
```

A divisão de responsabilidade é a seguinte: `acionador.py` responde **quanto
pedem**; `listagem_historico.py` responde **quanto tempo leva para sair, quanto
o preço cede no caminho e com que velocidade o estoque escoa**.

---

## 1. Como rodar

Credenciais vêm **só** do ambiente. Nenhum script tem senha embutida — crie o
`.env` (já ignorado pelo git) na raiz do projeto:

```
PGHOST=...
PGPORT=5432
PGDATABASE=coleta_imobiliaria
PGUSER=...
PGPASSWORD=...
```

Rotina normal — dois comandos, nenhum arquivo `.py` editado:

```powershell
.\coleta_semanal.ps1        # toda semana: DF e WI, cada um na sua pasta
python fluxo_mensal.py       # início do mês: ingestão + estudo + histórico
```

Etapas avulsas:

```powershell
# métricas de preço pedido — lote (todos os escopos com volume)
python analise/acionador/acionador.py

# histórico de listagem + métricas de mercado (12 meses)
python analise/acionador/listagem_historico.py --meses 12

# série mensal de um escopo, já construída
python analise/acionador/listagem_historico.py --somente-resumo --resumo "LAGO SUL" "CASA"

# testes do pipeline (base sintética, não toca no banco)
python analise/acionador/testes_pipeline.py

# conferência do que está no banco (somente leitura)
python analise/acionador/diagnostico_portais.py

# ingestão isolada
python BD/ingestao.py --dry-run
python BD/ingestao.py --arquivo dados/coletas/wi/2026-09-01.csv --portal wi
```

Além de `PG*`, o `.env` aceita `DATA_DIR` — a raiz das coletas (padrão
`./dados`). É o que permite cada máquina guardar os CSVs onde quiser sem que
isso vire alteração de código.

Uso programático:

```python
from acionador import EstudoMercado

em = EstudoMercado(bairro="LAGO NORTE", tipo="CASA CONDOMINIO")
em.carregar_dados()
em.enviar_banco_individual()
em.gerarResumo()

EstudoMercado().enviar_banco()          # lote, escopo descoberto no banco
EstudoMercado(auto_escopo=False).enviar_banco()   # lote na lista fixa antiga
```

---

## 2. O que mudou nesta rodada, e por quê

Dez correções de método. Todas mexem no número que sai no estudo, então estão
listadas com o efeito esperado. As três últimas (§2.8 a §2.10) vieram de
medição direta no banco, não de leitura de código.

### 2.1 Dedupe físico de anúncio
`dedupe_fisico=True` (padrão)

O `DISTINCT ON (codigo, data_coleta)` só resolvia repetição do **mesmo**
anúncio. O caso real é o imóvel na carteira de várias imobiliárias: entra com
códigos diferentes e pesa N vezes na mediana. Como anúncio muito compartilhado
se concentra em lançamento e alto padrão, o viés é **sistemático**, não ruído.

Assinatura: lat/lon a 4 casas (~11 m) + área arredondada em 2 m² + quartos +
preço em banda de 2%. Sobrevive o registro de menor preço. Registro sem
geolocalização passa intacto — sem geo não dá para afirmar que é o mesmo imóvel.

### 2.2 Colapso por listagem
`colapsar_por_listagem=True` (padrão)

A coleta é semanal. Sem colapsar, um anúncio parado 12 semanas entrava 12 vezes
na mediana e um que saiu em 1 semana entrava 1 vez — ou seja, **o encalhado
pesava mais**, e imóvel encalhado costuma ser o caro. O estudo lia o preço de
quem não vende.

Agora: uma linha por (código, mês), o último snapshot do mês.

**Consequência para quem lê o BI:** `amostra` mudou de significado — antes era
"linhas de snapshot", agora é "imóveis-mês". Os números caem de patamar (num
teste, 2841 → 816) sem que nada tenha sumido do mercado. A coluna nova
`imoveis_unicos` traz a contagem de códigos distintos.

### 2.3 Cauda: winsorização no lugar do IQR global
`metodo_outlier="winsor"` (padrão) · `"iqr_grupo"` · `"iqr"` · `"none"`

O IQR rodava sobre o bairro/tipo inteiro **antes** da segmentação, e
*deletava* a linha. No Lago Sul isso descartava justamente o ALTO LUXO — o
segmento que o estudo mais precisa. A winsorização trunca em p1/p99 e
**preserva** a linha: o imóvel de R$ 40M continua contando no segmento, no
cluster e na amostra, só não arrasta a média.

`iqr_grupo` é o meio-termo: IQR dentro de cada faixa de metragem, então um
sobrado de 900 m² deixa de ser outlier por ser grande.

Compatibilidade: `aplicar_iqr=False` continua desligando tudo;
`aplicar_iqr=True` explícito volta ao IQR global antigo.

### 2.4 Índice constant-quality (repeat-listing)
coluna nova `variacao_repeat_pct` + `n_repeat`

`variacao_m2_pct` compara a mediana de um mês com a do mês seguinte, e por isso
**mistura preço com composição**: um mês com mais imóveis de 4 quartos sobe a
mediana sem que preço nenhum tenha mudado. É o erro clássico de índice
imobiliário.

`variacao_repeat_pct` mede a variação **dentro do mesmo código**, de um mês
para o mês seguinte, e só depois agrega (mediana das variações individuais). Só
entram pares de meses consecutivos.

O teste sintético mostra a diferença com clareza — base construída com deriva
real de ~+0,4%/mês:

| mês | `variacao_m2_pct` (mix) | `variacao_repeat_pct` (real) |
|---|---|---|
| 2026-04 | -0,20% | +0,4% |
| 2026-05 | **-9,66%** | +0,2% |

Os -9,66% eram composição, não queda de preço. Esse é o número que ia para o
gráfico do estudo.

**Recomendação:** usar `variacao_repeat_pct` como a leitura de tendência e
manter `variacao_m2_pct` apenas como referência histórica.

### 2.5 Qualidade estatística no output
colunas novas `m2_p25`, `m2_p75`, `m2_desvio`, `imoveis_unicos`, `confiabilidade`

O gráfico desenhava -8% com 6 imóveis igual a -8% com 180. Agora cada linha traz
dispersão e um carimbo: `ALTA` (>= 30 imóveis), `MEDIA` (>= 15), `BAIXA` (>= 5),
`INSUFICIENTE`. Filtro direto no BI.

### 2.6 Escopo descoberto no banco
`auto_escopo=True` (padrão), `min_linhas_escopo=300`

`BAIRROS` tem 12 nomes fixos; o scraper já coleta ~50 bairros. O dado existia e
era descartado. O lote agora pergunta ao banco quais pares (bairro, tipo) têm
volume na janela e roda todos — bairro novo entra sozinho quando junta massa.

### 2.7 Credenciais, conexões e escrita
- Senha embutida removida de `acionador.py` e `BD/enviar_BD.py`. **A senha
  antiga está no histórico do git: precisa ser rotacionada no RDS.** Apagar do
  arquivo não basta.
- `with psycopg2.connect()` fecha a *transação*, não a conexão: o lote abria
  uma conexão por consulta (72+ escopos) e nunca as devolvia. Agora há uma
  conexão para o lote inteiro, via `closing()`.
- `_upsert`: um `DELETE` só para todos os combos (era um por combo) e payload
  vetorizado no lugar de `iterrows()` — que ainda devolvia escalares numpy,
  que o psycopg2 não adapta.
- `commit` por escopo: queda no meio do lote não descarta o que já rodou.

### 2.8 Dois portais na mesma tabela

O Wimóveis entrou na coleta e a tabela `imoveis` passou a receber duas fontes.
Três coisas quebravam em silêncio, e o diagnóstico rodado no banco em
**2026-09-01** confirmou cada uma.

**Chave de painel virou `(portal, id do anúncio)`.** `codigo` sozinho não
serve: é o código da imobiliária, repete entre portais, e — pior — está
corrompido em parte da base. Medido: `codigo` = `"Piso Frio"` em 1.663 linhas
de 47 bairros, `"Churrasqueira, Piscina"` em 617, parágrafos inteiros de
cláusula de fiança em 789. É texto de descrição caindo na coluna errada por
desalinhamento no scraper.

O identificador passou a sair do **fim da URL** (`link`), que está preenchido
em 100% das linhas recentes e termina no id do anúncio no portal:

```sql
upper(trim(coalesce(portal,'?'))) || '|' ||
coalesce(substring(link from '([0-9]{4,})/?$'), nullif(trim(codigo),''), 'ROW'||id::text)
```

Efeito medido numa janela de 2 meses: **61.092 imóveis distintos pela chave
nova contra 55.486 pelo `codigo`** — ou seja, o código estava fundindo cerca de
5.600 imóveis sem relação nenhuma, cada fusão contaminando mediana, DOM e
índice repeat.

**`ultima_coleta` passou a respeitar o filtro de elegibilidade.** Era calculada
sobre a tabela inteira; uma carga malformada avançava a data sem trazer
nenhum anúncio válido, e aí **todo** anúncio virava "inativo" — a base inteira
caía na conta de saídas e a absorção ia a 100%. Foi exatamente o que as cargas
de agosto/2026 provocaram: `ativos = 0`. Depois da correção, a mesma janela
devolve 51.250 ativos e DOM médio de 39,9 dias.

**`mercado_metricas` passou a agregar por portal.** O mesmo imóvel anunciado
nos dois portais são dois anúncios; somar contaria o estoque duas vezes.
Comparar portais é válido, somar não — enquanto não houver dedupe físico
entre eles.

### 2.8b Quadras da Asa Norte e Asa Sul agrupadas por centena

`SQN 409` vira `SQN 400`, `SQS 116` vira `SQS 100`. A série da centena é o
gradiente de valor do Plano Piloto — a 100 fica no Eixinho Oeste, a 400 cola na
W3 — e quadra a quadra a amostra não sustentava um ponto de gráfico.

O segmento `QUADRA_VAGA` nessas duas regiões era praticamente inútil antes:

| | antes (quadra crua) | depois (centena) |
|---|---|---|
| ASA NORTE venda | 171 linhas, 56 grupos — **zero** com confiança ALTA | 13 linhas, maioria ALTA |
| ASA SUL venda | 136 linhas, 45 grupos — 3 com confiança ALTA | 10 linhas, 8 ALTA |

E o resultado tem leitura urbana direta (agosto/2026, apartamento, com vaga):
SQN 100 R$ 14.647/m² · SQN 200 R$ 13.982 · SQN 300 R$ 13.714 · SQN 400
R$ 14.135 com vaga e R$ 11.440 sem — o degrau em direção à W3 aparece no número.

Configurado em `QUADRA_CENTENAS`, com lista de prefixos por bairro: prefixo
fora da lista (SEPN, SRTVS) ou quadra sem número cai em `""` e sai do segmento,
igual às faixas dos lagos. Cerca de 15-18% das quadras não trazem número.

### 2.9 Ingestão: o nome do arquivo saiu do código

`CARGAS` em `BD/enviar_BD.py` era reescrito toda carga, e essa edição virava
commit para limpar a cada `git pull`. `BD/ingestao.py` substitui a lista:

- **portal vem da pasta**, `dados/coletas/<portal>/` — acaba o `PORTAL = "df"`
  fixo que rotularia coleta do Wimóveis como DF Imóveis;
- **data vem do nome**, `AAAA-MM-DD.csv` — os dois scrapers já nomeiam assim
  (e escreviam por cima um do outro, porque usavam o mesmo default no mesmo
  diretório: era isso que obrigava a renomear na mão);
- **idempotência por hash**: `analytics.cargas` registra arquivo, portal, data
  e sha256. Rodar duas vezes não duplica;
- **`dados/` no `.gitignore`**, raiz configurável por `DATA_DIR` no `.env`.

E o contrato do dataframe fica num lugar só, documentado no cabeçalho de
`ingestao.py`: `portal`, `id_anuncio`, `codigo`, `oferta` (VENDA|ALUGUEL),
`tipo` (do imóvel), `data_coleta`, mais os numéricos. Qualquer layout de CSV
entra; só o canônico chega ao banco.

### 2.10 O layout dos scrapers mudou e ninguém avisou o banco

Achado mais caro do diagnóstico. Os scrapers hoje emitem:

```
tipo        = venda / aluguel / lancamento     <- isto é a OFERTA
tipo_imovel = apartamento / casa / ...         <- isto é o TIPO
```

O carregador continuou tratando `tipo` como tipo do imóvel e ignorando
`tipo_imovel`. O corte é limpo:

| período | estado |
|---|---|
| até 2026-08-13 | layout antigo, base saudável |
| de 2026-08-18 | 162.413 linhas com `oferta` vazia e `tipo` = VENDA/ALUGUEL |

Consequência: **as coletas de 18, 19, 20, 21, 26 e 28 de agosto estão
invisíveis para o estudo** — o acionador filtra `oferta in (VENDA, ALUGUEL,
PUBLICADO)` e nenhuma passa. No total, o acionador aproveitava **438.929 de
846.233 linhas (52%)** dos últimos 90 dias.

`BD/ingestao.py` resolve daqui para frente. O que já está no banco tem duas
classes, e `BD/corrigir_layout.py` trata as duas (relatório por padrão, grava
só com `--aplicar`):

| classe | linhas | reparo |
|---|---|---|
| A — `oferta` e `tipo` trocados entre si | ~45,6 mil | `--swap`, nada se perde |
| B — `oferta` vazia, `tipo` = oferta | ~162,4 mil | **recarregar os CSVs**; o `tipo_imovel` nunca chegou ao banco e não há como reconstruí-lo de lá |

Para a classe B, `--recuperar-oferta` é paliativo: salva a oferta e marca
`tipo` como nulo, devolvendo as linhas ao histórico de DOM/absorção (que não
filtra tipo) sem inventar um tipo que não existe. O reparo de verdade é
recarregar aqueles seis CSVs com `BD/ingestao.py`.

---

## 3. O que `listagem_historico.py` acrescenta

Reconstrói a vida de cada anúncio a partir do painel semanal que **já existe** —
sem coleta nova.

`analytics.listagem_historico` (uma linha por código x oferta): `primeira_vez`,
`ultima_vez`, `dias_no_ar`, `preco_inicial`, `preco_final`, `n_reducoes`,
`n_aumentos`, `variacao_preco_pct`, `ativo`, `censurado_esq`.

`analytics.mercado_metricas` (bairro x tipo x oferta x mês):

| métrica | leitura de negócio |
|---|---|
| `dom_mediano` | quanto tempo o imóvel fica no ar antes de sair |
| `absorcao_pct` | % do estoque do mês que escoou |
| `meses_estoque` | em quantos meses o estoque atual escoa no ritmo do mês |
| `pct_com_reducao` | % do estoque que já baixou o preço pelo menos uma vez |
| `desconto_mediano_pct` | quanto o mercado devolve entre quem baixou |
| `entradas` / `saidas` | pressão de oferta e escoamento |
| `cobertura` | REGULAR / IRREGULAR — ver abaixo |

**`cobertura` não é detalhe.** Bairro que o scraper visita esporadicamente
produz absorção altíssima e DOM baixíssimo sem que nada tenha acontecido no
mercado. Caso real medido: NORTE/CASA/VENDA tinha 2 a 7 imóveis em todo mês e
**113 só em julho/2026** — 107 entradas, 106 saídas, "93,8% de absorção,
DOM de 10 dias". Era o scraper passando ali uma única vez, e o bairro liderava
o ranking de velocidade.

O carimbo compara o estoque do mês com a **mediana da própria série** (a média
seria puxada justamente pelo pico): fora da faixa de 0,33× a 3×, ou série com
menos de 3 meses, vira `IRREGULAR`. `ranking_absorcao()` já filtra só
`REGULAR`.

`desconto_mediano_pct` e `dom_mediano` são os dois números que sustentam a
conversa de precificação com proprietário: deixam de ser opinião do corretor e
viram estatística do próprio bairro.

**Limite honesto, e ele é importante:** *sumir do portal* **não é** *vendeu*.
Pode ser venda, contrato expirado, anúncio pausado ou o mesmo imóvel
republicado com outro código. Enquanto o `Estoque/` (captação e saída da 61)
não for cruzado, isto é **proxy** de absorção — serve para comparar bairros e
acompanhar tendência, não para afirmar volume de vendas em número absoluto.

---

## 4. Pendências conhecidas

- **Recarregar as coletas de 18 a 28/08/2026** (162,4 mil linhas, classe B do
  §2.10). Só isso devolve o tipo do imóvel dessas semanas.
- **Duplicatas históricas:** 34.785 chaves `(portal, codigo, data_coleta)`
  repetidas, 98.608 linhas, pior caso 143×. Por isso `BD/ingestao.py` cria
  índice comum e **não** único — a criação do único falharia. Ver
  `python BD/ingestao.py --relatorio-duplicatas`.
- **12,4% das linhas sem latitude** — o dedupe físico (§2.1) não alcança essas.
- **Cruzamento com `Estoque/`** — é o único dado transacional real do
  repositório e continua fora do estudo. Fecha o gap pedido-vs-realizado e dá
  share da 61 por bairro.
- **Cluster de condição segue circular.** O KMeans agrupa por `valor_m2` e o
  relatório mostra `valor_m2` por grupo: sempre vai dar diferença, e ela não é
  informação nova. Enquanto não houver condição de verdade (texto do anúncio,
  ver §5.5), o rótulo "Original/Reformado/Nova" é uma **faixa de preço** com
  nome de reforma, e assim deve ser lido.
- **Sem histórico de versão das métricas.** Mudar `METRAGEM_BINS` ou
  `CLUSTER_CONFIG` reescreve o passado (`delete` + `insert`). Não dá para
  auditar o que mudou depois de um ajuste. Correção barata: coluna
  `versao_config` com o hash da configuração e parar de deletar versões antigas.
- **`enviar_banco.py` (693 linhas) é um fork antigo do mesmo pipeline** e não
  recebeu nenhuma destas correções. Enquanto existir, corre o risco de alguém
  rodar a versão velha. Recomendado apagar.
- **Amostra pequena em segmentos finos.** Com `QUADRA_VAGA` x `luxo` x
  `vaga_cat` o `n` cai rápido; a coluna `confiabilidade` expõe isso, mas a
  segmentação em si merece revisão.

---

## 5. Roadmap com máquina melhor

O que hoje não roda por hardware, o que cada coisa entrega e o que ela exige.
Ordem sugerida — cada item é útil sozinho, e os primeiros são os que mudam mais
a conversa comercial.

### 5.1 AVM — modelo hedônico de avaliação (o primeiro a fazer)

**Hoje:** o estudo entrega mediana de segmento. Para precificar um imóvel
específico o corretor procura "a linha mais parecida" na tabela.

**Com modelo:** gradient boosting (LightGBM/XGBoost/CatBoost) sobre
`log(valor_m2)`, com features `area_util`, `quartos`, `vagas`, `tipo`,
`bairro`, `quadra`, `lat/lon`, `mes_ref`. Entrega estimativa condicional para
**um** imóvel — não para o segmento dele.

Duas saídas que valem mais que o ponto central:
- **Regressão quantílica** (p10 / p50 / p90): faixa de preço defensável, não
  número solto. "Este imóvel está entre 8,9k e 11,4k/m²; você quer pedir 13k."
- **SHAP:** decomposição do valor por atributo. É o material da reunião com o
  proprietário — mostra quanto a vaga extra, a quadra e os 20 m² a mais valem
  em reais, com base no mercado dele.

**Hardware:** treino por bairro cabe em qualquer notebook. O que exige máquina é
o DF inteiro com tuning sério (Optuna, centenas de trials, validação temporal):
32 GB+ de RAM e horas de CPU, ou LightGBM/XGBoost em GPU. Retreino mensal.

**Validação obrigatória:** *walk-forward* temporal (treina até o mês M, testa em
M+1). Validação cruzada aleatória vaza o futuro e infla a métrica.

### 5.2 Modelo de sobrevivência para tempo de mercado

O item de maior valor comercial da lista.

`dias_no_ar` tem **censura à direita**: anúncio ainda ativo não terminou o
tempo dele, e média simples subestima o DOM. É problema de análise de
sobrevivência, não de média.

Com `scikit-survival` (Cox, Random Survival Forest, Gradient Boosting de
sobrevivência) sobre o painel: **probabilidade de sair em 30/60/90 dias dado o
preço pedido**. Isso vira a curva preço × tempo:

> "A R$ 1,45M, 62% de chance de sair em 90 dias. A R$ 1,65M, 24%."

É exatamente a conversa que hoje se faz por intuição.

**Hardware:** RSF é caro em memória — a matriz de risco cresce com o quadrado
das observações em alguns estimadores. Com 100k+ anúncios, 32-64 GB.

### 5.3 Índice de preço hedônico (aposenta o repeat-listing)

`variacao_repeat_pct` (§2.4) só usa anúncios presentes em dois meses seguidos —
descarta a maior parte da base. O **time-dummy hedonic index** usa todos: regride
`log(preço)` contra as características **e** dummies de mês; o coeficiente do mês
é a variação de preço com qualidade constante.

Índice de mercado publicável por bairro, sem efeito de composição. Com janela
móvel e reconciliação hierárquica (bairro → RA → DF), vira produto de imagem —
"Índice 61 de Preços de Brasília".

**Hardware:** regressão com muitas dummies em 1M+ linhas quer RAM; `pyfixest` /
`fixest` resolvem efeitos fixos altos sem estourar memória.

### 5.4 Camada geoespacial

Bairro é uma unidade grossa: no Lago Sul, a QI 5 e a QI 27 não são o mesmo
mercado — as `QUADRA_FAIXAS` no código já são uma tentativa manual de contornar
isso.

Com máquina: features de vizinhança (KNN espacial de preço, densidade de oferta
no raio, distância a eixos), indexação H3, e superfície contínua de preço
(kriging ou GWR) → **mapa de calor de R$/m² em célula de 200 m**, no lugar da
tabela por bairro. Substitui as faixas de quadra escritas à mão por estrutura
aprendida do dado.

**Hardware:** kriging é O(n³) na forma ingênua — é aqui que RAM e CPU aparecem.

### 5.5 NLP no texto do anúncio (mata o cluster circular)

O rótulo "Original / Reformado / Nova" é inferido de preço, o que o torna
circular (§4). Condição de verdade está escrita no anúncio: *reformado*,
*porteira fechada*, *a reformar*, *mobiliado*, *vista livre*, *andar alto*,
*nascente*.

Caminhos, do barato ao pesado:
1. regex e dicionário — resolve boa parte, roda em qualquer máquina;
2. rotular alguns milhares de anúncios com LLM (API Claude) e treinar um
   classificador leve em cima — não exige GPU local;
3. embeddings (`sentence-transformers`) ou LLM local quantizado (7-8B) para
   extração estruturada em volume — GPU de 12-16 GB.

**Pré-requisito:** o scraper não guarda título nem descrição hoje. É um campo a
mais no CSV e na tabela, e vale começar a coletar **antes** de ter a máquina —
NLP só funciona sobre histórico acumulado.

### 5.6 Deduplicação por aprendizado

O dedupe de hoje é regra fixa (§2.1) e depende de geo. Um modelo de *record
linkage* probabilístico (`splink`, `dedupe`) ou similaridade de embedding de
texto com blocking espacial pega o que a regra perde: anúncio sem coordenada,
preço divergente entre imobiliárias, área digitada errado.

**Hardware:** comparação par-a-par explode sem blocking; com blocking, roda bem
em CPU e ganha muito com paralelismo.

### 5.7 Previsão e cenário

Com `mercado_metricas` acumulado, previsão 3-6 meses de R$/m², absorção e DOM
por bairro. Baseline ETS/Prophet contra gradient boosting com features de
calendário e macro (Selic, IPCA, IGP-M, crédito imobiliário). Reconciliação
hierárquica mantém bairro coerente com RA e com o DF.

Entrega planejamento de captação: onde o estoque vai apertar, onde vai sobrar.

### 5.8 Efeito causal da redução de preço

O painel semanal registra reduções e saídas — dá para estimar o **efeito** de
baixar X% sobre a probabilidade de saída, com *uplift modeling* ou
diferenças-em-diferenças, em vez da correlação simples (que é enganosa: quem
baixa preço já é, em média, quem tinha imóvel mais difícil).

Resposta direta a "baixar 5% adianta?".

### 5.9 Visão computacional nas fotos

Padrão de acabamento e estado de conservação a partir da imagem (ViT/ResNet
com fine-tune). É o item mais pesado e o de retorno mais incerto: exige GPU,
armazenamento das imagens e coleta que o scraper hoje não faz. **Último da
fila.**

### 5.10 O que precisa existir antes de qualquer modelo pesado

Sem isto, máquina melhor só troca um número duvidoso por outro mais caro:

- **Dataset versionado** — congelar o recorte de treino, senão nenhum resultado
  é reprodutível.
- **Backtest walk-forward** como critério único de aceite. Nada entra em
  produção por R² de validação aleatória.
- **Baseline honesto.** O modelo novo tem que ganhar da mediana de segmento
  atual em MAE/MAPE fora da amostra. Muitas vezes não ganha, e saber disso vale
  mais do que subir o modelo.
- **Monitoramento de drift** — mercado muda, modelo azeda em silêncio.
- **Registro de qual versão gerou qual número** (§4), senão não há como
  explicar uma virada no gráfico.

### Resumo por exigência de máquina

| Item | Precisa de máquina melhor? |
|---|---|
| 5.1 AVM por bairro | não — pode começar hoje |
| 5.5 NLP via regra/API | não — falta só coletar o texto |
| 5.3 Índice hedônico | RAM |
| 5.2 Sobrevivência (DOM) | RAM |
| 5.6 Dedupe probabilístico | CPU paralela |
| 5.1 AVM DF inteiro + tuning | RAM + CPU/GPU |
| 5.4 Geoespacial contínuo | RAM + CPU |
| 5.7 Previsão hierárquica | CPU |
| 5.5 NLP com LLM local | GPU 12-16 GB |
| 5.9 Visão computacional | GPU + armazenamento |
