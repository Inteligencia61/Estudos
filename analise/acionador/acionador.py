from __future__ import annotations

import os
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2 import sql as psql


# ============================================================
# Helpers de data (mantidos fora da classe por serem utilitários puros)
# ============================================================

def _first_day_of_month(ym: str) -> date:
    y, m = map(int, ym.split("-"))
    return date(y, m, 1)


def _last_day_of_month(ym: str) -> date:
    d1 = _first_day_of_month(ym)
    if d1.month == 12:
        d2 = date(d1.year + 1, 1, 1)
    else:
        d2 = date(d1.year, d1.month + 1, 1)
    return (pd.Timestamp(d2) - pd.Timedelta(days=1)).date()


def _add_months(d: date, n: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=n)).date()


def _janela_3_meses(ym: str) -> Tuple[date, date]:
    fim = _last_day_of_month(ym)
    inicio = _add_months(_first_day_of_month(ym), -2)
    return inicio, fim


def criar_data(dt: date, ano: int, mes: int, qtd_ant: int) -> date:
    """Retorna a data do mês anterior ao informado, navegando anos se necessário."""
    max_mes = 12
    mes_calc = mes - qtd_ant
    if mes_calc < 1:
        ano -= 1
        mes_calc = max_mes + mes_calc  # mes_calc é negativo, então subtrai
    return date(ano, mes_calc, 1)


# ============================================================
# Classe principal
# ============================================================

class EstudoMercado:
    """
    Encapsula todo o pipeline de estudo de mercado imobiliário:
      - Conexão com banco
      - Carregamento e limpeza de dados
      - Clusterização (padrão e luxo)
      - Construção de métricas long
      - Upsert no banco
      - Visualização e resumo

    Uso individual (um bairro/tipo):
        em = EstudoMercado(bairro="ASA SUL", tipo="CASA")
        em.enviar_banco_individual()

    Uso em lote (todos os bairros e tipos):
        em = EstudoMercado()
        em.enviar_banco()
    """

    # ----------------------------------------------------------
    # Listas e configurações padrão
    # ----------------------------------------------------------
    BAIRROS = [
        "ASA SUL", "ASA NORTE", "NOROESTE", "SUDOESTE",
        "LAGO SUL", "LAGO NORTE", "JARDIM BOTANICO",
        "ARNIQUEIRA", "SUL", "NORTE", "ADE", "AREAL",
    ]
    TIPOS = ["CASA", "APARTAMENTO", "CASA CONDOMINIO"]
    BAIRROS_LUXO = ["LAGO SUL", "LAGO NORTE"]
    OFERTAS = ["VENDA", "ALUGUEL"]

    # Filtros de preço/m² por tipo de oferta (aplicados no envio em lote)
    FILTROS_OFERTA: Dict[str, Dict] = {
        "VENDA":   {"preco_min": 500_000, "preco_max": 50_000_000, "vlm2_min": 1_000, "vlm2_max": 900_000},
        "ALUGUEL": {"preco_min": 500,     "preco_max": 100_000,    "vlm2_min": 5,     "vlm2_max": 5_000},
    }

    # O topo era um balde único ">1000", onde vive quase todo o ALTO LUXO —
    # sobravam poucas linhas no segmento METRAGEM_VAGA. Dividido em
    # 1000-1500 / 1500-2000 / 2000-3000 / >3000 (03/08/2026).
    # As faixas abaixo de 1000 não mudaram: imóvel nenhum troca de faixa fora do topo.
    METRAGEM_LABELS = ["<75", "75-90", "90-130", "130-160", "160-200",
                       "200-400", "400-600", "600-800", "800-1000",
                       "1000-1500", "1500-2000", "2000-3000", ">3000"]
    METRAGEM_BINS   = [0, 75, 90, 130, 160, 200, 400, 600, 800, 1000,
                       1500, 2000, 3000, 10_000_000]

    # ----------------------------------------------------------
    # Configuração do cluster de condição (Original / Reformado / Nova)
    # ----------------------------------------------------------
    # Padrão histórico: KMeans k=9 sobre [valor_m2, area_util], com os 9 centros
    # ordenados por valor_m2 e agrupados 3 a 3.
    CLUSTER_FEATURES  = ["valor_m2", "area_util"]
    CLUSTER_N_PADRAO  = 9

    # Override por bairro. Só o LAGO NORTE usa a configuração nova (03/08/2026):
    # k=3 sobre apenas `valor_m2`.
    #   - `area_util` fazia o KMeans separar por TAMANHO, não por condição: os
    #     outliers de área capturavam um centro inteiro e deixavam um dos três
    #     rótulos residual (0,4%).
    #   - com k=9 + 2 features, remover ~0,2% das linhas da base reatribuía
    #     ~20% dos imóveis de rótulo (instabilidade entre rodadas).
    #   - k=3 sobre valor_m2: distribuição 25/51/24 e instabilidade 0,7%.
    # Os demais bairros seguem no padrão para não exigir regravação em massa.
    # A primeira feature é a usada para ordenar os centros e atribuir o rótulo.
    CLUSTER_CONFIG: Dict[str, Dict[str, Any]] = {
        "LAGO NORTE": {"n_clusters": 3, "features": ["valor_m2"]},
    }

    # Identifica uma "série" dentro de uma janela de mês-alvo: é o conjunto de
    # pontos (um por mes_ref) que forma UMA linha no gráfico de evolução.
    # Tudo menos `mes_ref` e as métricas.
    CHAVE_SERIE = ["segmento", "vaga_cat", "cluster_nome",
                   "quartos", "metragem_fx", "quadra", "luxo"]

    # ----------------------------------------------------------
    # Regras específicas dos lagos (LAGO SUL e LAGO NORTE)
    # ----------------------------------------------------------

    # Bairros que usam o conjunto de regras dos lagos:
    #   - faixas fixas de preço para luxo (em vez do KMeans)
    #   - agrupamento de quadras QI/QL em Início / Meio / Final
    #   - nomenclatura simplificada de cluster (Original / Reformado / Nova)
    BAIRROS_LAGOS = ["LAGO SUL", "LAGO NORTE"]

    # Faixas fixas de preço para classificação de luxo (substitui o KMeans
    # nos bairros de BAIRROS_LAGOS).
    LUXO_FAIXAS_LAGOS: Dict[str, float] = {
        "luxo_min": 2_500_000,      # a partir daqui já é LUXO
        "alto_luxo_min": 8_000_000, # acima disso é ALTO LUXO
    }

    # Agrupamento de quadras QI/QL em Início / Meio / Final.
    # As faixas são POR BAIRRO porque a numeração é diferente em cada lago:
    #   - LAGO SUL  (SHIS): QI ímpar 1-29 · QL par 2-28
    #   - LAGO NORTE (SHIN): QI e QL de 1 a 16, com pares e ímpares
    # Quadra cujo número não esteja em nenhuma faixa vira "" e sai do
    # segmento QUADRA_VAGA — por isso as faixas precisam cobrir toda a
    # numeração existente no bairro.
    QUADRA_FAIXAS: Dict[str, Dict[str, Dict[str, set]]] = {
        "LAGO SUL": {
            "QI": {
                "Início": {1, 3, 5},
                "Meio":   {7, 9, 11, 13, 15, 16},
                "Final":  {17, 19, 21, 23, 25, 26, 27, 28, 29},
            },
            "QL": {
                "Início": {2, 4, 6},
                "Meio":   {8, 10, 14, 16},
                "Final":  {18, 20, 22, 24, 26, 28},
            },
        },
        # LAGO NORTE — divisão informada pelo especialista do bairro.
        # Lado par e lado ímpar são geograficamente equivalentes e caem no
        # mesmo bloco:
        #   Início -> ímpar 1-5   · par 2-6
        #   Meio   -> ímpar 7-9   · par 8-10
        #   Final  -> ímpar 11-15 · par 12-16
        # Mesma divisão para QI e QL.
        "LAGO NORTE": {
            "QI": {
                "Início": {1, 2, 3, 4, 5, 6},
                "Meio":   {7, 8, 9, 10},
                "Final":  {11, 12, 13, 14, 15, 16},
            },
            "QL": {
                "Início": {1, 2, 3, 4, 5, 6},
                "Meio":   {7, 8, 9, 10},
                "Final":  {11, 12, 13, 14, 15, 16},
            },
        },
    }

    def __init__(
        self,
        bairro: Optional[str] = None,
        tipo: Optional[str] = None,
        oferta: str = "VENDA",
        meses_alvo: Optional[List[str]] = None,
        # filtros de preço/área
        preco_min: int = 500_000,
        preco_max: int = 50_000_000,
        area_min: int = 40,
        area_max: int = 1_500_000,
        vlm2_min: int = 1_000,
        vlm2_max: int = 900_000,
        aplicar_iqr: bool = True,
        # cluster
        clusters_ativos: bool = True,
        # Fallback: usado nos bairros sem entrada em CLUSTER_CONFIG.
        kmeans_n_clusters: int = 9,
        random_state: int = 42,
        min_amostra_cluster: int = 10,
        min_amostra_cluster_luxo: int = 12,
        # banco
        tabela_fonte: str = "imoveis",
        schema_analytics: str = "analytics",
        tbl_metricas: str = "estudo_metricas",
        upsert_page_size: int = 2000,
        min_amostra_segmento: int = 5,
        # True  = corte por série (mantém a linha do gráfico de evolução)
        # False = corte por ponto (comportamento anterior)
        corte_por_serie: bool = True,
    ):
        self.data = date.today()

        # escopo do estudo
        self.bairro_unico = bairro.strip().upper() if bairro else None
        self.tipo_unico   = tipo.strip().upper()   if tipo   else None
        self.oferta       = oferta.strip().upper()

        # meses-alvo: se não informado, usa os 3 meses anteriores ao atual
        if meses_alvo:
            self.meses_alvo = meses_alvo
        else:
            self.meses_alvo = self._meses_anteriores(3)

        # filtros
        self.preco_min   = preco_min
        self.preco_max   = preco_max
        self.area_min    = area_min
        self.area_max    = area_max
        self.vlm2_min    = vlm2_min
        self.vlm2_max    = vlm2_max
        self.aplicar_iqr = aplicar_iqr

        # cluster
        self.clusters_ativos          = clusters_ativos
        self.kmeans_n_clusters        = kmeans_n_clusters
        self.random_state             = random_state
        self.min_amostra_cluster      = min_amostra_cluster
        self.min_amostra_cluster_luxo = min_amostra_cluster_luxo

        # banco
        self.tabela_fonte      = tabela_fonte
        self.schema_analytics  = schema_analytics
        self.tbl_metricas      = tbl_metricas
        self.upsert_page_size  = upsert_page_size
        self.min_amostra_segmento = min_amostra_segmento
        self.corte_por_serie      = corte_por_serie

        # estado interno
        self.df_analisado: Optional[pd.DataFrame] = None   # dados brutos carregados
        self.df_bf: Optional[pd.DataFrame]        = None   # métricas long geradas

    # ----------------------------------------------------------
    # Helpers internos
    # ----------------------------------------------------------

    def _meses_anteriores(self, n: int) -> List[str]:
        """Retorna os n meses anteriores ao mês atual no formato YYYY-MM."""
        meses = []
        for i in range(n, 0, -1):
            dt = criar_data(self.data, self.data.year, self.data.month, i)
            meses.append(dt.strftime("%Y-%m"))
        return meses

    def _pg_connect(self):
        return psycopg2.connect(
            host=os.getenv("PGHOST", "db-restore.ctug6oqcsj14.us-east-2.rds.amazonaws.com"),
            port=int(os.getenv("PGPORT", "5432")),
            dbname=os.getenv("PGDATABASE", "coleta_imobiliaria"),
            user=os.getenv("PGUSER", "inteligencia"),
            password=os.getenv("PGPASSWORD", "61imoveis"),
        )

    def _ensure_schema_and_table(self, conn) -> None:
        schema = self.schema_analytics
        tbl    = self.tbl_metricas
        raw    = self.tabela_fonte

        ddl = f"""
        create schema if not exists {schema};

        create table if not exists {schema}.{tbl} (
          bairro         text not null,
          tipo           text not null,
          oferta         text not null,
          mes_alvo       text not null,
          janela_inicio  date not null,
          janela_fim     date not null,
          mes_ref        text not null,
          segmento       text not null,
          vaga_cat       text not null,
          cluster_nome   text not null default '',
          quartos        int  not null default -1,
          metragem_fx    text not null default '',
          quadra         text not null default '',
          luxo           text not null default '',
          amostra        int not null,
          m2_medio       double precision,
          m2_mediana     double precision,
          preco_mediana  double precision,
          area_mediana   double precision,
          variacao_m2_pct double precision,
          gerado_em      timestamp not null default now(),
          primary key (
            bairro, tipo, oferta,
            mes_alvo, janela_inicio, janela_fim,
            mes_ref, segmento,
            vaga_cat, cluster_nome, quartos, metragem_fx, quadra, luxo
          )
        );

        create index if not exists idx_{tbl}_filtros
          on {schema}.{tbl} (bairro, tipo, oferta, mes_alvo, segmento, mes_ref);

        create index if not exists idx_{raw}_data_coleta on {raw} (data_coleta);
        create index if not exists idx_{raw}_filtros_data on {raw} (bairro, tipo, oferta, data_coleta);
        create index if not exists idx_{raw}_codigo_data  on {raw} (codigo, data_coleta);
        """
        with conn.cursor() as cur:
            cur.execute(ddl)
        with conn.cursor() as cur:
            cur.execute(f"""
                alter table {schema}.{tbl}
                add column if not exists luxo text not null default '';
                alter table {schema}.{tbl}
                add column if not exists variacao_m2_pct double precision;
            """)

    # ----------------------------------------------------------
    # Carregamento de dados
    # ----------------------------------------------------------

    def _carregar_do_banco(self, bairro: str, tipo: str,
                           inicio: date, fim: date) -> pd.DataFrame:
        oferta_alvo   = self.oferta
        ofertas_aceitas = list({oferta_alvo, "PUBLICADO"})

        q = psql.SQL("""
            SELECT DISTINCT ON (codigo, data_coleta)
                TRIM(codigo)::text                      as codigo,
                UPPER(TRIM(bairro))::text               as bairro,
                UPPER(TRIM(cidade))::text               as cidade,
                UPPER(TRIM(tipo))::text                 as tipo,
                UPPER(TRIM(oferta))::text               as oferta,
                area_util::double precision             as area_util,
                preco::double precision                 as preco,
                quartos::double precision               as quartos,
                vagas::double precision                 as vagas,
                latitude::double precision              as latitude,
                longitude::double precision             as longitude,
                UPPER(TRIM(quadra))::text               as quadra,
                data_coleta::date                       as data_coleta
            FROM {tabela}
            WHERE data_coleta >= %s
              AND data_coleta <= %s
              AND UPPER(TRIM(bairro)) = %s
              AND UPPER(TRIM(tipo)) = %s
              AND UPPER(TRIM(oferta)) = ANY(%s)
              AND preco is not null
              AND area_util is not null
              AND preco >= %s AND preco <= %s
              AND area_util >= %s AND area_util <= %s
            ORDER BY codigo, data_coleta, data_coleta DESC;
        """).format(tabela=psql.Identifier(self.tabela_fonte))

        with self._pg_connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(q, (
                    inicio, fim, bairro, tipo, ofertas_aceitas,
                    self.preco_min, self.preco_max,
                    self.area_min,  self.area_max,
                ))
                rows = cur.fetchall()

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce").dt.date
        for c in ["preco", "area_util", "quartos", "vagas", "latitude", "longitude"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["bairro", "cidade", "tipo", "oferta", "quadra", "codigo"]:
            if c in df.columns:
                df[c] = df[c].astype("string").str.strip().str.upper()
        return df

    # ----------------------------------------------------------
    # Limpeza
    # ----------------------------------------------------------

    def _remover_outliers_iqr(self, df: pd.DataFrame, coluna: str) -> pd.DataFrame:
        s = df[coluna].dropna()
        if s.empty:
            return df
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return df
        return df[(df[coluna] >= q1 - 1.5 * iqr) & (df[coluna] <= q3 + 1.5 * iqr)].copy()

    def _bairro_usa_regra_lagos(self, bairro: str) -> bool:
        return str(bairro).strip().upper() in {b.upper() for b in self.BAIRROS_LAGOS}

    def _mapear_quadra_lagos(self, quadra: Optional[str], bairro: str) -> str:
        """
        Agrupa as quadras dos lagos (LAGO SUL e LAGO NORTE) em blocos
        Início / Meio / Final, separadamente para QI e QL, usando as faixas do
        bairro em QUADRA_FAIXAS. Quadras fora das faixas (ou não identificadas)
        retornam string vazia, igual ao comportamento padrão.
        """
        faixas = self.QUADRA_FAIXAS.get(str(bairro).strip().upper())
        if not faixas or not quadra:
            return ""

        m = re.search(r"\bQ([IL])\s*0*?(\d+)\b", str(quadra).upper())
        if not m:
            return ""

        prefixo = "Q" + m.group(1)   # "QI" ou "QL"
        numero  = int(m.group(2))

        for bloco, numeros in faixas.get(prefixo, {}).items():
            if numero in numeros:
                return f"{prefixo} - {bloco}"
        return ""

    def _limpar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().dropna(subset=["preco", "area_util"])
        df["valor_m2"] = df["preco"] / df["area_util"]
        df = df[(df["valor_m2"] >= self.vlm2_min) & (df["valor_m2"] <= self.vlm2_max)]

        if self.aplicar_iqr and len(df) >= 20:
            df = self._remover_outliers_iqr(df, "valor_m2")

        df["quartos"]  = pd.to_numeric(df["quartos"], errors="coerce").fillna(0).astype(int)
        df["vagas"]    = pd.to_numeric(df["vagas"],   errors="coerce").fillna(0).astype(int)
        df["data_dt"]  = pd.to_datetime(df["data_coleta"], errors="coerce")
        df["mes_ref"]  = df["data_dt"].dt.to_period("M").astype("string")
        df["vaga_cat"] = np.where(df["vagas"] > 0, "COM VAGA", "SEM VAGA")

        df["metragem_fx"] = pd.cut(
            df["area_util"],
            bins=self.METRAGEM_BINS,
            labels=self.METRAGEM_LABELS,
            include_lowest=True,
            right=False,
        ).astype("string").fillna("")

        df["quadra"] = (
            df["quadra"].astype("string").fillna("").str.strip().str.upper()
            if "quadra" in df.columns else ""
        )

        # LAGO SUL / LAGO NORTE: substitui a quadra bruta pelo agrupamento
        # QI/QL Início/Meio/Final
        if "bairro" in df.columns and df["bairro"].notna().any():
            bairro_ref = str(df["bairro"].dropna().iloc[0]).strip().upper()
            if self._bairro_usa_regra_lagos(bairro_ref):
                df["quadra"] = (
                    df["quadra"]
                    .apply(lambda q: self._mapear_quadra_lagos(q, bairro_ref))
                    .astype("string")
                    .fillna("")
                )

        df["luxo"] = ""
        return df

    # ----------------------------------------------------------
    # Clusterização padrão
    # ----------------------------------------------------------

    def _cluster_feats(self, bairro: str) -> List[str]:
        """Features do cluster de condição para o bairro (ver CLUSTER_CONFIG)."""
        cfg = self.CLUSTER_CONFIG.get(str(bairro).strip().upper(), {})
        return list(cfg.get("features", self.CLUSTER_FEATURES))

    def _cluster_k(self, bairro: str) -> int:
        """Nº de centros do KMeans para o bairro (ver CLUSTER_CONFIG)."""
        cfg = self.CLUSTER_CONFIG.get(str(bairro).strip().upper(), {})
        return int(cfg.get("n_clusters", self.kmeans_n_clusters))

    def _treinar_cluster_global(self, df_treino: pd.DataFrame, bairro: str = ""):
        if not SKLEARN_OK:
            raise RuntimeError("sklearn indisponível.")
        if len(df_treino) < self.min_amostra_cluster:
            raise RuntimeError("Amostra insuficiente para cluster global.")

        feats = self._cluster_feats(bairro)
        base  = df_treino.dropna(subset=feats)
        if len(base) < self.min_amostra_cluster:
            raise RuntimeError("Amostra insuficiente (pós dropna) para cluster global.")

        scaler = StandardScaler()
        X = scaler.fit_transform(base[feats])
        km = KMeans(n_clusters=self._cluster_k(bairro),
                    random_state=self.random_state, n_init=10)
        km.fit(X)

        centers_real = scaler.inverse_transform(km.cluster_centers_)
        order = pd.DataFrame(centers_real, columns=feats).sort_values("valor_m2").index.tolist()

        # LAGO SUL / LAGO NORTE usam a nomenclatura simplificada
        # Original / Reformado / Nova
        if self._bairro_usa_regra_lagos(bairro):
            labels = ["Original", "Reformado", "Nova"]
        else:
            labels = ["01 - Original", "02 - Semi-Reformado", "03 - Reformado"]

        # Distribui os centros (já ordenados por valor_m2) nos 3 rótulos,
        # independente de kmeans_n_clusters:
        #   k=3 -> 1 centro por rótulo
        #   k=9 -> 3 centros por rótulo (comportamento anterior)
        n = len(order)
        mapping = {cid: labels[min(i * 3 // n, 2)] for i, cid in enumerate(order)}
        return scaler, km, mapping

    def _aplicar_cluster_fixo(self, dados: pd.DataFrame, scaler, km, mapping,
                              bairro: str = "") -> pd.DataFrame:
        # Precisa usar exatamente as mesmas features do treino, senão o
        # scaler/predict recebe shape diferente.
        feats = self._cluster_feats(bairro)
        dfc   = dados.dropna(subset=feats).copy()
        if dfc.empty:
            return dfc
        X = scaler.transform(dfc[feats])
        dfc["cluster_id"]   = km.predict(X)
        dfc["cluster_nome"] = dfc["cluster_id"].map(mapping).astype("string").fillna("")
        return dfc

    # ----------------------------------------------------------
    # Clusterização luxo
    # ----------------------------------------------------------

    def _bairro_e_luxo(self, bairro: str) -> bool:
        return bairro.strip().upper() in {b.upper() for b in self.BAIRROS_LUXO}

    def _aplicar_faixa_luxo_lagos(self, dados: pd.DataFrame) -> pd.DataFrame:
        """
        LAGO SUL e LAGO NORTE não usam KMeans para luxo: usam faixas fixas de preço.
          - preco < 2.5M              -> "" (não entra na faixa de luxo)
          - 2.5M <= preco <= 8M       -> "LUXO"
          - preco > 8M                -> "ALTO LUXO"
        """
        df = dados.copy()
        if df.empty:
            df["luxo"] = ""
            return df

        luxo_min      = self.LUXO_FAIXAS_LAGOS["luxo_min"]
        alto_luxo_min = self.LUXO_FAIXAS_LAGOS["alto_luxo_min"]

        condicoes = [
            df["preco"] > alto_luxo_min,
            df["preco"] >= luxo_min,
        ]
        escolhas = ["ALTO LUXO", "LUXO"]
        df["luxo"] = np.select(condicoes, escolhas, default="")
        df["luxo"] = df["luxo"].astype("string").fillna("")
        return df

    def _aplicar_cluster_luxo(self, dados: pd.DataFrame) -> pd.DataFrame:
        df = dados.copy()
        if df.empty:
            df["luxo"] = ""
            return df

        bairro_ref = (
            str(df["bairro"].dropna().iloc[0]).strip().upper()
            if "bairro" in df.columns and df["bairro"].notna().any() else ""
        )

        # LAGO SUL / LAGO NORTE: faixas fixas de preço (sem KMeans)
        if self._bairro_usa_regra_lagos(bairro_ref):
            return self._aplicar_faixa_luxo_lagos(df)

        # Demais bairros de luxo continuam no KMeans antigo (hoje BAIRROS_LUXO
        # e BAIRROS_LAGOS coincidem, então este caminho fica inativo)
        if not self._bairro_e_luxo(bairro_ref) or not SKLEARN_OK:
            df["luxo"] = ""
            return df

        colunas = ["preco", "valor_m2", "area_util"]
        base    = df.dropna(subset=colunas).copy()
        if len(base) < self.min_amostra_cluster_luxo:
            df["luxo"] = ""
            return df

        scaler = StandardScaler()
        X = scaler.fit_transform(base[colunas])
        km = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        base["cluster_luxo_id"] = km.fit_predict(X)

        order = (
            base.groupby("cluster_luxo_id")["preco"]
            .mean().sort_values().index.tolist()
        )
        mapping = {order[0]: "LUXO", order[1]: "SUPER LUXO"}
        base["luxo"] = base["cluster_luxo_id"].map(mapping).astype("string").fillna("")

        df["luxo"] = ""
        df.loc[base.index, "luxo"] = base["luxo"]
        return df

    # ----------------------------------------------------------
    # Agregação de métricas
    # ----------------------------------------------------------

    def _agg_metricas(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby(group_cols, dropna=False)
            .agg(
                amostra      =("valor_m2", "size"),
                m2_medio     =("valor_m2", "mean"),
                m2_mediana   =("valor_m2", "median"),
                preco_mediana=("preco",    "median"),
                area_mediana =("area_util","median"),
            )
            .reset_index()
        )

    def _build_metricas_long(
        self,
        dados_janela: pd.DataFrame,
        mes_alvo: str,
        inicio: date,
        fim: date,
        bairro: str,
        tipo: str,
        scaler=None, km=None, mapping=None,
    ) -> pd.DataFrame:
        if dados_janela.empty:
            return pd.DataFrame()

        base = dados_janela.copy()
        base = base.dropna(subset=["mes_ref", "vaga_cat", "valor_m2", "preco", "area_util"])
        if base.empty:
            return pd.DataFrame()

        base = self._aplicar_cluster_luxo(base)
        base["luxo"] = base["luxo"].astype("string").fillna("")

        linhas = []

        # 1) GERAL_VAGA
        g = self._agg_metricas(base, ["mes_ref", "vaga_cat", "luxo"])
        if not g.empty:
            g["segmento"] = "GERAL_VAGA"
            g["cluster_nome"] = g["metragem_fx"] = g["quadra"] = ""
            g["quartos"] = -1
            linhas.append(g)

        # 2) CLUSTER_VAGA
        if self.clusters_ativos and scaler is not None and km is not None:
            dfc = self._aplicar_cluster_fixo(base, scaler, km, mapping, bairro)
            if not dfc.empty:
                dfc["luxo"] = dfc["luxo"].astype("string").fillna("")
                c = self._agg_metricas(dfc, ["mes_ref", "vaga_cat", "cluster_nome", "luxo"])
                if not c.empty:
                    c["segmento"] = "CLUSTER_VAGA"
                    c["metragem_fx"] = c["quadra"] = ""
                    c["quartos"] = -1
                    linhas.append(c)

        # 3) QUARTOS_VAGA
        qbase = base[base["quartos"] > 0].copy()
        if not qbase.empty:
            q = self._agg_metricas(qbase, ["mes_ref", "vaga_cat", "quartos", "luxo"])
            if not q.empty:
                q["segmento"] = "QUARTOS_VAGA"
                q["cluster_nome"] = q["metragem_fx"] = q["quadra"] = ""
                linhas.append(q)

        # 4) METRAGEM_VAGA
        mbase = base[base["metragem_fx"].astype(str).str.len() > 0].copy()
        if not mbase.empty:
            m = self._agg_metricas(mbase, ["mes_ref", "vaga_cat", "metragem_fx", "luxo"])
            if not m.empty:
                m["segmento"] = "METRAGEM_VAGA"
                m["cluster_nome"] = m["quadra"] = ""
                m["quartos"] = -1
                linhas.append(m)

        # 5) QUADRA_VAGA
        qd = base[base["quadra"].astype(str).str.len() > 0].copy()
        if not qd.empty:
            qq = self._agg_metricas(qd, ["mes_ref", "vaga_cat", "quadra", "luxo"])
            if not qq.empty:
                qq["segmento"] = "QUADRA_VAGA"
                qq["cluster_nome"] = qq["metragem_fx"] = ""
                qq["quartos"] = -1
                linhas.append(qq)

        if not linhas:
            return pd.DataFrame()

        out = pd.concat(linhas, ignore_index=True)

        if self.corte_por_serie:
            # Corte POR SÉRIE, não por ponto: a combinação é mantida em todos os
            # mes_ref da janela se atingir min_amostra_segmento em ALGUM mês.
            # Sem isso, um mês fraco apaga o ponto e quebra a linha do gráfico de
            # evolução (ex.: ALTO LUXO/Original com amostra 1 e 4 nos meses
            # anteriores e 7 no mês-alvo aparecia só no mês-alvo).
            # Os pontos fracos continuam identificáveis pela coluna `amostra`.
            passa = (
                out.groupby(self.CHAVE_SERIE, dropna=False)["amostra"]
                .transform("max") >= self.min_amostra_segmento
            )
            out = out[passa].copy()
        else:
            out = out[out["amostra"] >= self.min_amostra_segmento].copy()

        if out.empty:
            return out

        out["bairro"]        = bairro
        out["tipo"]          = tipo
        out["oferta"]        = self.oferta
        out["mes_alvo"]      = mes_alvo
        out["janela_inicio"] = inicio
        out["janela_fim"]    = fim

        for col in ["cluster_nome", "metragem_fx", "quadra", "luxo"]:
            out[col] = out.get(col, "").astype("string").fillna("")
        out["quartos"] = pd.to_numeric(out.get("quartos", -1), errors="coerce").fillna(-1).astype(int)

        out = self._add_variacao_mensal(out)

        return out[[
            "bairro", "tipo", "oferta",
            "mes_alvo", "janela_inicio", "janela_fim",
            "mes_ref", "segmento",
            "vaga_cat", "cluster_nome", "quartos", "metragem_fx", "quadra", "luxo",
            "amostra", "m2_medio", "m2_mediana", "preco_mediana", "area_mediana",
            "variacao_m2_pct",
        ]]

    def _add_variacao_mensal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula variação % de m2_mediana entre meses consecutivos dentro do mesmo segmento."""
        if df.empty:
            df["variacao_m2_pct"] = pd.NA
            return df

        group_keys = [
            "bairro", "tipo", "oferta", "mes_alvo", "segmento",
            "vaga_cat", "cluster_nome", "quartos", "metragem_fx", "quadra", "luxo",
        ]
        df = df.sort_values(group_keys + ["mes_ref"]).copy()
        df["variacao_m2_pct"] = (
            df.groupby(group_keys, dropna=False)["m2_mediana"]
            .pct_change()
            .mul(100)
            .round(2)
        )
        return df

    # ----------------------------------------------------------
    # Upsert
    # ----------------------------------------------------------

    def _upsert(self, conn, df_long: pd.DataFrame) -> None:
        if df_long.empty:
            return

        schema = self.schema_analytics
        tbl    = self.tbl_metricas

        # Remove linhas existentes para o escopo sendo reprocessado antes de inserir,
        # evitando dependência de constraint específico no ON CONFLICT.
        combos = df_long[["bairro", "tipo", "oferta", "mes_alvo"]].drop_duplicates()
        del_q = psql.SQL(
            "delete from {schema}.{tbl} "
            "where bairro = %s and tipo = %s and oferta = %s and mes_alvo = %s"
        ).format(schema=psql.Identifier(schema), tbl=psql.Identifier(tbl))
        with conn.cursor() as cur:
            for _, row in combos.iterrows():
                cur.execute(del_q, (row["bairro"], row["tipo"], row["oferta"], row["mes_alvo"]))

        cols = [
            "bairro", "tipo", "oferta",
            "mes_alvo", "janela_inicio", "janela_fim",
            "mes_ref", "segmento",
            "vaga_cat", "cluster_nome", "quartos", "metragem_fx", "quadra", "luxo",
            "amostra", "m2_medio", "m2_mediana", "preco_mediana", "area_mediana",
            "variacao_m2_pct",
        ]
        payload = [
            tuple(r[c] if pd.notna(r[c]) else None for c in cols)
            for _, r in df_long.iterrows()
        ]
        q = psql.SQL("""
            insert into {schema}.{tbl} (
              bairro, tipo, oferta,
              mes_alvo, janela_inicio, janela_fim,
              mes_ref, segmento,
              vaga_cat, cluster_nome, quartos, metragem_fx, quadra, luxo,
              amostra, m2_medio, m2_mediana, preco_mediana, area_mediana,
              variacao_m2_pct
            ) values %s
        """).format(schema=psql.Identifier(schema), tbl=psql.Identifier(tbl))

        with conn.cursor() as cur:
            execute_values(cur, q.as_string(conn), payload,
                           page_size=self.upsert_page_size)

    # ----------------------------------------------------------
    # Pipeline interno reutilizável
    # ----------------------------------------------------------

    def _rodar_pipeline(self, bairro: str, tipo: str) -> pd.DataFrame:
        """
        Executa o pipeline completo para um bairro+tipo e retorna o df_long.
        Não grava no banco — apenas processa e retorna.
        """
        janelas = [(ym, *_janela_3_meses(ym)) for ym in self.meses_alvo]
        inicio_global = min(x[1] for x in janelas)
        fim_global    = max(x[2] for x in janelas)

        df_raw = self._carregar_do_banco(bairro, tipo, inicio_global, fim_global)
        if df_raw.empty:
            print(f"[INFO] Sem dados: {bairro} / {tipo}")
            return pd.DataFrame()

        df_limpo = self._limpar_dados(df_raw)
        if df_limpo.empty:
            print(f"[INFO] Sem dados válidos após limpeza: {bairro} / {tipo}")
            return pd.DataFrame()

        scaler = km = mapping = None
        if self.clusters_ativos:
            try:
                scaler, km, mapping = self._treinar_cluster_global(df_limpo, bairro)
            except Exception as e:
                print(f"[WARN] Cluster desativado para {bairro}/{tipo}: {e}")

        partes = []
        for (ym, ini, fim) in janelas:
            janela = df_limpo[
                (df_limpo["data_coleta"] >= ini) &
                (df_limpo["data_coleta"] <= fim)
            ].copy()
            if janela.empty:
                continue
            df_long = self._build_metricas_long(
                janela, ym, ini, fim, bairro, tipo, scaler, km, mapping
            )
            if not df_long.empty:
                partes.append(df_long)

        return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

    # ----------------------------------------------------------
    # API pública
    # ----------------------------------------------------------

    def enviar_banco(self) -> None:
        """
        Envia métricas para TODOS os bairros, tipos e ofertas (VENDA e ALUGUEL).
        """
        with self._pg_connect() as conn:
            self._ensure_schema_and_table(conn)
            for oferta in self.OFERTAS:
                self.oferta = oferta
                filtros = self.FILTROS_OFERTA.get(oferta, {})
                for k, v in filtros.items():
                    setattr(self, k, v)
                for bairro in self.BAIRROS:
                    for tipo in self.TIPOS:
                        df_long = self._rodar_pipeline(bairro, tipo)
                        if df_long.empty:
                            continue
                        self._upsert(conn, df_long)
                        print(f"[OK] {oferta} / {bairro} / {tipo}: {len(df_long)} linhas gravadas.")
            conn.commit()
        print("Envio em lote concluído.")

    def enviar_banco_individual(self) -> None:
        """
        Envia métricas para o bairro e tipo definidos no construtor.
        Requer que bairro_unico e tipo_unico sejam informados.
        """
        if not self.bairro_unico or not self.tipo_unico:
            raise ValueError("Informe bairro e tipo no construtor para usar enviar_banco_individual().")

        df_long = self._rodar_pipeline(self.bairro_unico, self.tipo_unico)
        if df_long.empty:
            print("Nenhuma métrica gerada.")
            return

        self.df_bf = df_long

        with self._pg_connect() as conn:
            self._ensure_schema_and_table(conn)
            self._upsert(conn, df_long)
            conn.commit()
        print(f"[OK] {self.bairro_unico} / {self.tipo_unico}: {len(df_long)} linhas gravadas.")

    def ver_dados(self) -> Optional[pd.DataFrame]:
        """
        Retorna o df de métricas gerado na última execução do pipeline.
        Execute enviar_banco_individual() ou carregar_dados() antes.
        """
        if self.df_bf is None:
            print("[INFO] Nenhum dado disponível. Execute o pipeline primeiro.")
        return self.df_bf

    def carregar_dados(self) -> pd.DataFrame:
        """
        Carrega e limpa os dados do banco (sem gravar métricas).
        Armazena em self.df_analisado e retorna o df.
        """
        if not self.bairro_unico or not self.tipo_unico:
            raise ValueError("Informe bairro e tipo no construtor.")

        janelas = [(ym, *_janela_3_meses(ym)) for ym in self.meses_alvo]
        inicio  = min(x[1] for x in janelas)
        fim     = max(x[2] for x in janelas)

        df_raw = self._carregar_do_banco(self.bairro_unico, self.tipo_unico, inicio, fim)
        self.df_analisado = self._limpar_dados(df_raw) if not df_raw.empty else df_raw
        return self.df_analisado

    def gerarResumo(self) -> None:
        """
        Exibe resumo estatístico do df_bf (métricas long geradas).
        Execute enviar_banco_individual() antes.
        """
        if self.df_bf is None or self.df_bf.empty:
            print("Sem dados para resumir.")
            return

        print("=" * 60)
        print(f"  RESUMO — {self.bairro_unico} / {self.tipo_unico}")
        print("=" * 60)
        print(f"  Meses-alvo : {self.meses_alvo}")
        print(f"  Total de linhas: {len(self.df_bf)}")
        print()

        for mes in self.df_bf["mes_alvo"].unique():
            sub = self.df_bf[self.df_bf["mes_alvo"] == mes]
            print(f"  Mês-alvo: {mes}  ({len(sub)} linhas)")
            for seg in sub["segmento"].unique():
                s = sub[sub["segmento"] == seg]
                print(f"    [{seg}]  amostra={s['amostra'].sum()}  "
                      f"m2_med={s['m2_mediana'].mean():.0f}  "
                      f"preco_med={s['preco_mediana'].mean():.0f}")
        print("=" * 60)

    def gerarGraficoCluster(self) -> None:
        """
        Gera gráfico de dispersão dos clusters no df_analisado.
        Execute carregar_dados() antes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[ERRO] matplotlib não instalado. pip install matplotlib")
            return

        if self.df_analisado is None or self.df_analisado.empty:
            print("Sem dados. Execute carregar_dados() antes.")
            return

        df = self.df_analisado.copy()

        # tenta criar cluster para visualização
        scaler = km = mapping = None
        if self.clusters_ativos and SKLEARN_OK:
            try:
                scaler, km, mapping = self._treinar_cluster_global(df, self.bairro_unico or "")
                df = self._aplicar_cluster_fixo(df, scaler, km, mapping, self.bairro_unico or "")
            except Exception as e:
                print(f"[WARN] Não foi possível gerar clusters para o gráfico: {e}")

        col_cor = "cluster_nome" if "cluster_nome" in df.columns else None

        fig, ax = plt.subplots(figsize=(9, 6))
        if col_cor:
            for nome, grp in df.groupby(col_cor):
                ax.scatter(grp["area_util"], grp["valor_m2"], label=nome, alpha=0.6, s=20)
            ax.legend(title="Cluster", fontsize=8)
        else:
            ax.scatter(df["area_util"], df["valor_m2"], alpha=0.6, s=20)

        ax.set_xlabel("Área útil (m²)")
        ax.set_ylabel("Valor/m² (R$)")
        titulo = f"Cluster — {self.bairro_unico or 'Todos'} / {self.tipo_unico or 'Todos'}"
        ax.set_title(titulo)
        plt.tight_layout()
        plt.show()


# ============================================================
# Helpers de data (mantidos para compatibilidade com código existente)
# ============================================================

dt = date.today()
dt_1ant = criar_data(dt, dt.year, dt.month, 1)
dt_2ant = criar_data(dt, dt.year, dt.month, 2)
dt_3ant = criar_data(dt, dt.year, dt.month, 3)

dt_1 = dt_1ant.strftime('%Y-%m')
dt_2 = dt_2ant.strftime('%Y-%m')
dt_3 = dt_3ant.strftime('%Y-%m')


# ============================================================
# Exemplos de uso
# ============================================================
if __name__ == "__main__":

    # --- Individual: um bairro e tipo específicos ---
    em = EstudoMercado(bairro="LAGO NORTE", tipo="CASA CONDOMINIO")
    em.carregar_dados()
    em.enviar_banco_individual()
    em.gerarResumo()
    em.gerarGraficoCluster()
    print(em.ver_dados())

    # --- Lote: todos os bairros e tipos ---
    #em = EstudoMercado()
    #em.enviar_banco()

    # --- Só carregar dados, sem gravar ---
    #em = EstudoMercado(bairro="ASA NORTE", tipo="APARTAMENTO")
    #df = em.carregar_dados()
    #em.gerarGraficoCluster()
    #somatorio = 0
    #for a in df['valor_m2']:
    #    somatorio += a

    #print(somatorio)
    #print(somatorio / len(df["valor_m2"]))