from __future__ import annotations

import os
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

    METRAGEM_LABELS = ["<75", "75-90", "90-130", "130-160", "160-200",
                       "200-400", "400-600", "600-800", "800-1000", ">1000"]
    METRAGEM_BINS   = [0, 75, 90, 130, 160, 200, 400, 600, 800, 1000, 10_000_000]

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
        kmeans_n_clusters: int = 9,
        random_state: int = 42,
        min_amostra_cluster: int = 10,
        min_amostra_cluster_luxo: int = 12,
        # banco
        tabela_fonte: str = "imoveis",
        schema_analytics: str = "analytics",
        tbl_metricas: str = "estudo_metricas",
        upsert_page_size: int = 2000,
        min_amostra_segmento: int = 3,
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
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
            dbname=os.getenv("PGDATABASE", "postgres"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", ""),
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
            """)

    # ----------------------------------------------------------
    # Carregamento de dados
    # ----------------------------------------------------------

    def _carregar_do_banco(self, bairro: str, tipo: str,
                           inicio: date, fim: date) -> pd.DataFrame:
        oferta_alvo   = self.oferta
        ofertas_aceitas = list({oferta_alvo, "PUBLICADO", "VENDA"})

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
        df["luxo"] = ""
        return df

    # ----------------------------------------------------------
    # Clusterização padrão
    # ----------------------------------------------------------

    def _treinar_cluster_global(self, df_treino: pd.DataFrame):
        if not SKLEARN_OK:
            raise RuntimeError("sklearn indisponível.")
        if len(df_treino) < self.min_amostra_cluster:
            raise RuntimeError("Amostra insuficiente para cluster global.")

        feats = ["valor_m2", "area_util"]
        base  = df_treino.dropna(subset=feats)
        if len(base) < self.min_amostra_cluster:
            raise RuntimeError("Amostra insuficiente (pós dropna) para cluster global.")

        scaler = StandardScaler()
        X = scaler.fit_transform(base[feats])
        km = KMeans(n_clusters=self.kmeans_n_clusters,
                    random_state=self.random_state, n_init=10)
        km.fit(X)

        centers_real = scaler.inverse_transform(km.cluster_centers_)
        order = pd.DataFrame(centers_real, columns=feats).sort_values("valor_m2").index.tolist()
        labels  = ["01 - Original", "02 - Semi-Reformado", "03 - Reformado"]
        mapping = {cid: labels[min(i // 3, 2)] for i, cid in enumerate(order)}
        return scaler, km, mapping

    def _aplicar_cluster_fixo(self, dados: pd.DataFrame, scaler, km, mapping) -> pd.DataFrame:
        feats = ["valor_m2", "area_util"]
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

    def _aplicar_cluster_luxo(self, dados: pd.DataFrame) -> pd.DataFrame:
        df = dados.copy()
        if df.empty:
            df["luxo"] = ""
            return df

        bairro_ref = (
            str(df["bairro"].dropna().iloc[0]).strip().upper()
            if "bairro" in df.columns and df["bairro"].notna().any() else ""
        )
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
            dfc = self._aplicar_cluster_fixo(base, scaler, km, mapping)
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

        return out[[
            "bairro", "tipo", "oferta",
            "mes_alvo", "janela_inicio", "janela_fim",
            "mes_ref", "segmento",
            "vaga_cat", "cluster_nome", "quartos", "metragem_fx", "quadra", "luxo",
            "amostra", "m2_medio", "m2_mediana", "preco_mediana", "area_mediana",
        ]]

    # ----------------------------------------------------------
    # Upsert
    # ----------------------------------------------------------

    def _upsert(self, conn, df_long: pd.DataFrame) -> None:
        if df_long.empty:
            return

        schema = self.schema_analytics
        tbl    = self.tbl_metricas
        cols   = [
            "bairro", "tipo", "oferta",
            "mes_alvo", "janela_inicio", "janela_fim",
            "mes_ref", "segmento",
            "vaga_cat", "cluster_nome", "quartos", "metragem_fx", "quadra", "luxo",
            "amostra", "m2_medio", "m2_mediana", "preco_mediana", "area_mediana",
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
              amostra, m2_medio, m2_mediana, preco_mediana, area_mediana
            ) values %s
            on conflict do nothing
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
                scaler, km, mapping = self._treinar_cluster_global(df_limpo)
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
        Envia métricas para TODOS os bairros e tipos definidos em BAIRROS e TIPOS.
        """
        with self._pg_connect() as conn:
            self._ensure_schema_and_table(conn)
            for bairro in self.BAIRROS:
                for tipo in self.TIPOS:
                    df_long = self._rodar_pipeline(bairro, tipo)
                    if df_long.empty:
                        continue
                    self._upsert(conn, df_long)
                    print(f"[OK] {bairro} / {tipo}: {len(df_long)} linhas gravadas.")
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
                scaler, km, mapping = self._treinar_cluster_global(df)
                df = self._aplicar_cluster_fixo(df, scaler, km, mapping)
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
    #em = EstudoMercado(bairro="ASA SUL", tipo="CASA")
    #em.carregar_dados()
    #em.enviar_banco_individual()
    #em.gerarResumo()
    #em.gerarGraficoCluster()
    #print(em.ver_dados())

    # --- Lote: todos os bairros e tipos ---
    #em = EstudoMercado()
    #em.enviar_banco()

    # --- Só carregar dados, sem gravar ---
    em = EstudoMercado(bairro="NOROESTE", tipo="APARTAMENTO")
    df = em.carregar_dados()
    em.gerarGraficoCluster()
    print(df.tail())
