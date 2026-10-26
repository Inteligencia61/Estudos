# Benchmark — Estudo de Mercado do DFimoveis

Análise do relatório que o DFimoveis (via TIMIPRO) gera para anunciantes,
comparado com o que o nosso pipeline (`acionador.py` + `listagem_historico.py`)
já calcula. Objetivo: decidir o que copiar, o que evitar e o que construir.

Documento de referência analisado: *Estudo de Mercado — 1459676 — 61 Imóveis —
09/2026*, imóvel SMDB Conjunto 18, Lago Sul (casa em condomínio, 4 quartos,
600 m², R$ 12.000.000).

---

## 1. O que o relatório deles contém

PDF de 9 páginas, por imóvel, com a marca do corretor (61 Imóveis) na capa e na
contracapa.

| Pág. | Seção | Conteúdo |
|---|---|---|
| 1 | Capa | Endereço, tipologia, preço anunciado, R$/m² do imóvel, contato do corretor, nota "sobre este relatório" |
| 2 | Perfil da região | Texto descritivo, foto, habitantes e renda média do domicílio (PDAD 2021, CODEPLAN/IPEDF-DF), preço e área médios do segmento, "quem vive e quem procura" |
| 3 | Preços praticados | Preço médio (R$ 4,73 Mi), m² médio (R$ 6.300), nº de anúncios (22), área média (783 m²), m² do imóvel (R$ 20.000); série de 12 meses do m² do segmento com a linha do imóvel; faixa menor valor / este imóvel / maior valor |
| 4 | Demanda real | Leads em 12 meses (1.152), acessos aos anúncios (70.511), leads por oferta (52,36), leads por mês, origem do contato (Indicação WhatsApp 53,3%, WhatsApp 32,5%, Telefone 14,1%) |
| 5 | Valorização | Variação anual do m² mês a mês, valorização do segmento (+5,70%), do bairro (+9,29%), do último trimestre (-2,20%), ranking de bairros (Sig, Lago Sul, Taquari) |
| 6 | Liquidez | Tempo médio de anúncio dos imóveis que saíram (~10,4 meses) e dos ativos (~10,4 meses); valor médio de saída (R$ 4,18 Mi) comparado com o preço pedido |
| 7 | O imóvel | Fotos, área, quartos, suítes, vagas, parágrafo de posicionamento |
| 8 | Localização | Mapa OpenStreetMap com o pino |
| 9 | Contracapa | Contato do corretor, nota de fontes, recorte técnico ("recorte de preço 15761 · demanda 15761 · carga de 09/2026") |

Segmentação usada: cidade + bairro + tipo + padrão + quartos, janela de 12
meses.

---

## 2. O que vale copiar

### 2.1 O formato — peça entregável por imóvel

É o principal ganho. O estudo deles é um PDF pronto para mostrar ao
proprietário. O nosso termina em `analytics.estudo_metricas` /
`analytics.mercado_metricas` e no BI: os números existem, mas não há peça para
o cliente.

### 2.2 Blocos "Como ler"

Cada gráfico vem com uma caixa curta explicando a leitura (o que é lead, por
que a linha de valorização desce sem o imóvel desvalorizar etc.). Custo baixo
e evita interpretação errada pelo proprietário.

### 2.3 Perfil da região com dados públicos

Habitantes e renda média do domicílio por região administrativa (PDAD,
CODEPLAN/IPEDF-DF). Tabela estática, atualizada a cada nova PDAD. Dá contexto
e credibilidade ao relatório.

### 2.4 Faixa de preço visual

Menor valor / este imóvel / maior valor em barras horizontais. É o visual mais
legível do relatório. Podemos fazer melhor com p25 / mediana / p75 do
segmento, que já existem em `m2_p25` / `m2_p75`, em vez de mínimo e máximo
(sensíveis a um único anúncio).

### 2.5 Valor médio de saída comparado com o preço pedido

Já temos o insumo: `analytics.listagem_historico` traz `preco_final` dos
anúncios que deixaram de estar ativos.

### 2.6 Mapa e ranking de bairros

- Mapa estático OSM com o pino a partir de lat/lon do anúncio.
- Ranking de valorização entre bairros, filtrado por `cobertura = REGULAR` e
  `confiabilidade` mínima.

### 2.7 Demanda (leads, acessos, canal)

Só o portal tem esses dados; não dá para copiar. Equivalentes que temos ou
podemos ter:

- leads da própria 61 por segmento (CRM / Imoview), se exportáveis;
- `entradas` / `saidas` de `mercado_metricas` como proxy de pressão de oferta
  e escoamento.

---

## 3. Fragilidades do relatório deles

Pontos em que o nosso método já é mais sólido. Servem de argumento comercial
e de lista do que **não** repetir.

| # | Problema no relatório DFimoveis | Como tratamos |
|---|---|---|
| 1 | **Valorização mistura preço com composição.** Com 22 imóveis, "+5,70% no ano" e "-2,20% no trimestre" refletem provavelmente mudança no mix de anúncios, não de preço. | `variacao_repeat_pct`: variação dentro do mesmo anúncio, depois agregada (README §2.4). |
| 2 | **Sem indicação de confiabilidade.** 22 imóveis aparecem com a mesma segurança que 500. | Colunas `confiabilidade` (ALTA / MEDIA / BAIXA / INSUFICIENTE), `m2_p25`, `m2_p75`, `m2_desvio` (README §2.5). |
| 3 | **Ranking suspeito.** "Sig +23,10%" no topo tem cara de amostra pequena ou cobertura irregular. | `cobertura = IRREGULAR` e `ranking_absorcao()` filtrando só `REGULAR` (README §3). |
| 4 | **Liquidez com cara de bug.** "~10,4 meses" idêntico nos cards de imóveis que saíram e de anúncios ativos. Além disso, trata "saiu do portal" como venda provável. | `dom_mediano` separado de estoque ativo; limite "sumir ≠ vender" documentado (README §3). |
| 5 | **Falta o número que sustenta a precificação:** quantos baixaram o preço e quanto baixaram. | `pct_com_reducao` e `desconto_mediano_pct` em `mercado_metricas`. |
| 6 | **Leitura complacente.** Imóvel 218% acima do m² médio vira "destaca-se pela arquitetura colonial e amplo terreno". Não confronta o proprietário com o risco de encalhe num segmento que leva ~10 meses. | Podemos cruzar posição na faixa com DOM e desconto do segmento e dizer isso explicitamente. |
| 7 | **Gráfico de m² com o imóvel "acima da escala".** A linha do imóvel nem cabe no eixo; o gráfico não comunica a distância. | Eixo que inclui o imóvel, ou mostrar a distância em número ao lado do gráfico. |
| 8 | **Média de razões vs razão de médias.** R$ 4,73 Mi ÷ 783 m² ≈ R$ 6.040/m², mas o relatório mostra R$ 6.300/m². Não está errado (média dos m² individuais), mas não é explicado. | Usamos mediana de `valor_m2`; deixar explícito no "Como ler". |
| 9 | **Série de valorização curta.** O gráfico de "12 meses" mostra só abril a setembro (6 pontos). | Série mensal completa por escopo (`--somente-resumo --resumo`). |
| 10 | **Área sem definição.** "600 m² área útil" numa casa do Lago Sul; não fica claro se o R$/m² compara área construída com área construída. | Documentar qual área entra no `valor_m2` e filtrar terreno quando houver. |

Conferências que **batem** no relatório deles: soma dos leads mensais = 1.152;
1.152 ÷ 22 = 52,36 leads por oferta; canais somam 99,9% (arredondamento).

---

## 4. Proposta — gerador "Estudo de Mercado 61"

PDF por imóvel, lendo `analytics.estudo_metricas`,
`analytics.mercado_metricas` e `analytics.listagem_historico`. A estrutura
segue a deles, com os diferenciais abaixo.

### 4.1 Estrutura sugerida

1. **Capa** — imóvel, preço, R$/m², corretor da 61.
2. **Região** — PDAD (habitantes, renda), texto, mediana e área do segmento.
3. **Preço** — faixa p25 / mediana / p75 com o imóvel posicionado, série do
   m² do segmento, selo de confiabilidade.
4. **Liquidez e negociação** — DOM mediano, % do estoque com redução,
   desconto mediano, meses de estoque, valor médio de saída comparado com o
   preço pedido.
5. **Valorização** — índice repeat (`variacao_repeat_pct`), ranking de
   bairros com cobertura regular.
6. **Comparáveis** — 5 a 10 anúncios semelhantes (área, quartos, vagas,
   distância), com preço, R$/m², dias no ar e link.
7. **Demanda 61** — leads da 61 no segmento (se o CRM permitir).
8. **O imóvel e localização** — fotos, atributos, mapa OSM.
9. **Contracapa** — contato, fontes, data da carga, versão da configuração.

Cada seção com bloco "Como ler".

### 4.2 Diferenciais em relação ao DFimoveis

- Faixa p25–p75 e confiabilidade em vez de média solta.
- DOM mediano, % com redução e desconto mediano — base da conversa de
  precificação.
- Índice repeat no lugar de comparação de medianas.
- Lista de comparáveis concretos.
- Leitura honesta da posição do preço (risco de encalhe quando o imóvel está
  acima do p75 num segmento lento).
- Etapa seguinte: faixa de preço sugerida pelo AVM (README §5.1), com
  decomposição SHAP.

### 4.3 Decisões em aberto

- Identidade visual: layout próprio da 61 ou neutro.
- Origem dos dados do imóvel: Imoview ou entrada manual.
- Disponibilidade de exportação de leads do CRM por segmento.
- Fonte e frequência de atualização da tabela PDAD.
