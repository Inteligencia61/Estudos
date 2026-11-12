from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sqlite3
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dfimoveis_native import ChromeNativeTransport, NativeTransportError
from dfimoveis_checkpoint import CollectionCheckpoint, CheckpointError

if TYPE_CHECKING:
    from playwright.sync_api import Page


BASE_URL = "https://www.dfimoveis.com.br"
CSV_FIELDS = [
    "id",
    "link",
    "codigo",
    "creci",
    "anunciante",
    "tipo",
    "tipo_imovel",
    "area_util",
    "bairro",
    "cidade",
    "preco",
    "valor_m2",
    "quartos",
    "vagas",
    "latitude",
    "longitude",
    "quadra",
    "data",
]
BAIRROS_POR_CIDADE = {
    "brasilia": [
        "asa norte",
        "asa sul",
        "altiplano leste",
        "granja do torto",
        "jardim botanico",
        "jardins mangueiral",
        "lago norte",
        "lago sul",
        "noroeste",
        "octogonal",
        "park sul",
        "park way",
        "setor industrial",
        "setor tororo",
        "sig",
        "sudoeste",
        "taquari",
        "vila da telebrasilia",
        "vila planalto",
        "zona civico administrativa",
        "zona rural",
    ],
    "aguas claras": [
        "ade",
        "areal",
        "arniqueira",
        "norte",
        "sul",
    ],
}
CIDADES_SEM_BAIRRO = [
    "aguas lindas de goias",
    "alphaville",
    "brazlandia",
    "candangolandia",
    "ceilandia",
    "cidade ocidental",
    "cruzeiro",
    "formosa",
    "gama",
    "guara",
    "jardim botanico",
    "luziania",
    "nucleo bandeirante",
    "paranoa",
    "planaltina",
    "planaltina de goias",
    "riacho fundo",
    "samambaia",
    "santa maria",
    "santo antonio do descoberto",
    "sao sebastiao",
    "setor industrial",
    "sobradinho",
    "taguatinga",
    "valparaiso de goias",
    "varjao",
    "vicente pires",
    "vila estrutural",
]
CIDADE_ALIASES = {
    "aguas-lindas-de-goiais": "aguas lindas de goias",
    "jardim-botanico": "jardim botanico",
    "luziana": "luziania",
    "planaltina-de-goiais": "planaltina de goias",
    "sao-sebastiao": "sao sebastiao",
    "valparaiso-de-goiais": "valparaiso de goias",
}


@dataclass(frozen=True)
class ScrapeConfig:
    oferta: str
    inicio: int
    fim: int | None
    delay: float
    detalhes: bool
    timeout: int
    limite: int | None
    escopo: str
    cidades: list[str] | None
    bairros: list[str] | None
    retries: int = 3
    backoff: float = 2.0


@dataclass(frozen=True)
class ListingScope:
    cidade: str | None = None
    bairro: str | None = None

    @property
    def label(self) -> str:
        if self.cidade and self.bairro:
            return f"{self.cidade}/{self.bairro}"
        if self.cidade:
            return self.cidade
        return "todos"


def normalize_space(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return fix_mojibake(value).strip()


def fix_mojibake(value: str) -> str:
    if "Ã" not in value and "Â" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def only_digits(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\D+", "", value)


def slugify(value: str) -> str:
    value = normalize_space(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def normalize_city_key(value: str) -> str:
    city = normalize_space(value).lower()
    return CIDADE_ALIASES.get(slugify(city), city)


def parse_decimal_text(value: str | None) -> str:
    if not value:
        return ""
    value = normalize_space(value)
    match = re.search(r"[\d.,]+", value)
    if not match:
        return ""
    number = match.group(0)
    if "," in number:
        number = number.split(",", 1)[0]
    elif number.count(".") == 1 and len(number.rsplit(".", 1)[1]) <= 2:
        number = number.rsplit(".", 1)[0]
    return only_digits(number)


def extract_real_html(raw: str) -> str:
    """Handles HTML saved from Chrome's source viewer and normal page HTML."""
    source_soup = BeautifulSoup(raw, "html.parser")
    source_lines = source_soup.select("td.line-content")
    if not source_lines:
        return raw
    return "\n".join(html.unescape(cell.get_text()) for cell in source_lines)


def make_soup(raw: str) -> BeautifulSoup:
    return BeautifulSoup(extract_real_html(raw), "html.parser")


# Status que valem nova tentativa. 404 fica de fora de proposito: e o sinal de
# fim de paginacao, nao uma falha.
STATUS_RETENTAVEIS = {429, 500, 502, 503, 504}

# Codigo de saida quando o portal exige o desafio anti-robo da Cloudflare.
SAIDA_PORTAL_BLOQUEADO = 4


class PortalBloqueado(RuntimeError):
    """O portal respondeu com o desafio da Cloudflare ("Just a moment...").

    Desde 11/2026 o DF Imoveis responde 403 + `cf-mitigated: challenge` em
    todas as paginas, inclusive a inicial. Antes este 403 caia no retry
    generico: 4 tentativas por pagina, escopo a escopo, horas de execucao e
    um CSV vazio no fim com saida 0. Repetir nao resolve — a coleta para na
    primeira pagina.
    """


class ColetaIncompleta(RuntimeError):
    """A listagem nao foi coletada por completo."""


def _e_desafio_cloudflare(response: requests.Response) -> bool:
    if response.status_code not in (403, 503):
        return False
    if response.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    inicio = response.text[:5000].lower()
    return "just a moment" in inicio or "challenge-platform" in inicio


def fetch(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """GET com retry exponencial.

    O portal derruba conexao de vez em quando (read timeout). Sem retry, uma
    unica falha no meio de uma varredura de horas mata a execucao inteira.
    """
    ultimo_erro: Exception | None = None

    for tentativa in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if _e_desafio_cloudflare(response):
                raise PortalBloqueado(
                    f"{url} respondeu {response.status_code} com desafio da Cloudflare "
                    f"(cf-ray {response.headers.get('cf-ray', '?')})"
                )
            if response.status_code in STATUS_RETENTAVEIS:
                raise requests.HTTPError(
                    f"status {response.status_code}", response=response
                )
            response.raise_for_status()
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding or "utf-8"
            return response.text

        except requests.HTTPError as exc:
            # 404 sobe na hora: quem chama usa isso para parar o escopo.
            if exc.response is not None and exc.response.status_code == 404:
                raise
            ultimo_erro = exc
        except requests.RequestException as exc:
            ultimo_erro = exc

        if tentativa < retries:
            espera = backoff ** tentativa
            print(
                f"  falha ({type(ultimo_erro).__name__}); "
                f"tentativa {tentativa + 2}/{retries + 1} em {espera:.0f}s",
                file=sys.stderr,
            )
            time.sleep(espera)

    raise ultimo_erro


def fetch_chrome(page: Page, url: str, timeout: int) -> str:
    from playwright.sync_api import Error, TimeoutError

    timeout_ms = timeout * 1000
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        if response is None:
            raise requests.RequestException(f"Chrome nao recebeu resposta de {url}")
        if response.headers.get("cf-mitigated") == "challenge":
            print("Verificacao do DF Imoveis: conclua-a na janela do Chrome se solicitada.",
                  file=sys.stderr)
            try:
                response = page.wait_for_event(
                    "response",
                    predicate=lambda item: (
                        item.request.is_navigation_request()
                        and item.frame == page.main_frame
                        and item.url.split("#")[0] == url.split("#")[0]
                        and item.headers.get("cf-mitigated") != "challenge"
                    ),
                    timeout=timeout_ms,
                )
            except TimeoutError as exc:
                raise PortalBloqueado(f"Verificacao pendente em {url}") from exc
            page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        if response.status == 403:
            raise PortalBloqueado(f"{url} respondeu 403 no Chrome")
        if response.status >= 400:
            http_response = requests.Response()
            http_response.status_code = response.status
            raise requests.HTTPError(f"status {response.status} em {url}", response=http_response)
        return page.content()
    except Error as exc:
        raise requests.RequestException(f"Falha no Chrome em {url}: {exc}") from exc


def build_listing_url(oferta: str, pagina: int, scope: ListingScope | None = None) -> str:
    if scope and scope.cidade and scope.bairro:
        cidade = slugify(scope.cidade)
        bairro = slugify(scope.bairro)
        return f"{BASE_URL}/{oferta}/df/{cidade}/{bairro}/imoveis?pagina={pagina}"
    if scope and scope.cidade:
        cidade = slugify(scope.cidade)
        return f"{BASE_URL}/{oferta}/df/{cidade}/imoveis?pagina={pagina}"
    return f"{BASE_URL}/{oferta}/df/todos/imoveis?pagina={pagina}"


def parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    values = [normalize_space(item).lower() for item in value.split(",")]
    return [item for item in values if item]


def build_scopes(config: ScrapeConfig) -> list[ListingScope]:
    if config.escopo == "todos":
        return [ListingScope()]

    selected_cities = config.cidades or [*BAIRROS_POR_CIDADE, *CIDADES_SEM_BAIRRO]
    scopes: list[ListingScope] = []
    for cidade in selected_cities:
        city_key = normalize_city_key(cidade)
        if city_key not in BAIRROS_POR_CIDADE:
            if city_key not in CIDADES_SEM_BAIRRO:
                valid = ", ".join([*BAIRROS_POR_CIDADE, *CIDADES_SEM_BAIRRO])
                raise SystemExit(f"Cidade sem mapeamento: {cidade}. Cidades disponiveis: {valid}")
            if config.bairros:
                raise SystemExit(f"A cidade {cidade} esta configurada sem bairros. Remova --bairros para usa-la.")
            scopes.append(ListingScope(cidade=city_key))
            continue
        selected_bairros = config.bairros or BAIRROS_POR_CIDADE[city_key]
        for bairro in selected_bairros:
            scopes.append(ListingScope(cidade=city_key, bairro=normalize_space(bairro).lower()))
    return scopes


def text_of(element) -> str:
    return normalize_space(element.get_text(" ", strip=True)) if element else ""


def attr_of(element, name: str) -> str:
    return normalize_space(element.get(name)) if element and element.get(name) else ""


def parse_offer_type_area(description: str) -> tuple[str, str, str]:
    description = normalize_space(description)
    match = re.search(r"^(Venda|Aluguel)\s+(.+?)\s+([\d.,]+)\s*m", description, flags=re.I)
    if not match:
        parts = description.split()
        return (parts[0].lower() if parts else "", " ".join(parts[1:]).lower(), "")
    return match.group(1).lower(), match.group(2).strip().lower(), parse_decimal_text(match.group(3))


def parse_location(title: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in normalize_space(title).split(",") if part.strip()]
    quadra = parts[0] if parts else ""
    cidade = parts[-1] if len(parts) >= 2 else ""
    bairro = parts[-2] if len(parts) >= 3 else ""
    return bairro.upper(), cidade.upper(), quadra


def parse_card(article) -> dict[str, str]:
    link_element = article.select_one("a.imovel-card")
    href = attr_of(link_element, "href")
    link = urljoin(BASE_URL, href)

    listing_id = attr_of(article.select_one("[data-id]"), "data-id")
    if not listing_id:
        match = re.search(r"-(\d+)(?:\?|$)", link)
        listing_id = match.group(1) if match else ""

    title = text_of(article.select_one('[itemprop="name"]'))
    description = text_of(article.select_one("h3[itemprop='description']"))
    tipo_operacao, tipo_imovel, area_util = parse_offer_type_area(description)
    bairro, cidade, quadra = parse_location(title)

    price_element = article.select_one('[itemprop="price"]')
    preco = parse_decimal_text(attr_of(price_element, "content") or text_of(price_element))
    valor_m2 = ""
    price_box = article.select_one(".imovel-price")
    if price_box:
        match = re.search(r"Valor\s*m\S*\s*R\$\s*([\d.,]+)", text_of(price_box), flags=re.I)
        if match:
            valor_m2 = parse_decimal_text(match.group(1))

    features = [text_of(item) for item in article.select(".imovel-feature div")]
    quartos = find_feature_number(features, "Quarto")
    vagas = find_feature_number(features, "Vaga")
    if not area_util:
        area_util = find_area(features)

    anunciante = attr_of(article.select_one(".imovel-anunciante img"), "alt")
    creci = ""
    creci_label = article.find(string=re.compile(r"Creci", re.I))
    if creci_label:
        next_text = creci_label.find_parent().find_next("p") if hasattr(creci_label, "find_parent") else None
        creci = text_of(next_text)

    return {
        "id": listing_id,
        "link": link,
        "codigo": "",
        "creci": creci,
        "anunciante": anunciante,
        "tipo": tipo_operacao,
        "tipo_imovel": tipo_imovel,
        "area_util": area_util,
        "bairro": bairro,
        "cidade": cidade,
        "preco": preco,
        "valor_m2": valor_m2,
        "quartos": quartos,
        "vagas": vagas,
        "latitude": "",
        "longitude": "",
        "quadra": quadra,
        "data": date.today().isoformat(),
    }


def find_feature_number(features: Iterable[str], keyword: str) -> str:
    for feature in features:
        if keyword.lower() in feature.lower():
            match = re.search(r"\d+", feature)
            return match.group(0) if match else ""
    return ""


def find_area(features: Iterable[str]) -> str:
    for feature in features:
        if re.search(r"m\s*(2|²|Â²)", feature, flags=re.I):
            return parse_decimal_text(feature)
    return ""


def parse_listing(raw: str) -> list[dict[str, str]]:
    soup = make_soup(raw)
    articles = soup.select('article[itemtype="https://schema.org/RealEstateListing"]')
    return [parse_card(article) for article in articles if article.select_one("a.imovel-card")]


def parse_detail(raw: str) -> dict[str, str]:
    real_html = extract_real_html(raw)
    soup = BeautifulSoup(real_html, "html.parser")
    detail: dict[str, str] = {}

    filtro_match = re.search(r"window\.imovelFiltro\s*=\s*(\{.*?\});", real_html, flags=re.S)
    if filtro_match:
        try:
            filtro = json.loads(filtro_match.group(1))
            detail.update(
                {
                    "id": str(filtro.get("IdExterno") or ""),
                    "tipo": normalize_space(str(filtro.get("Negocio") or "")).lower(),
                    "tipo_imovel": normalize_space(str(filtro.get("Tipo") or "")).lower(),
                    "bairro": normalize_space(str(filtro.get("Bairro") or "")).upper(),
                    "cidade": normalize_space(str(filtro.get("Cidade") or "")).upper(),
                    "quartos": only_digits(str(filtro.get("Quartos") or "")),
                    "preco": parse_decimal_text(str(filtro.get("Valor") or "")),
                }
            )
        except json.JSONDecodeError:
            pass

    detail["latitude"] = regex_group(real_html, r"\blatitude\s*=\s*([-\d.]+)")
    detail["longitude"] = regex_group(real_html, r"\blongitude\s*=\s*([-\d.]+)")

    code_label = soup.find(string=re.compile(r"C[oó]digo", re.I))
    if code_label:
        parent = code_label.find_parent("li") or code_label.find_parent()
        detail["codigo"] = normalize_space(parent.get_text(" ", strip=True)).split(":")[-1].strip()

    detail["area_util"] = parse_decimal_text(text_of(soup.select_one('[itemprop="floorSize"]'))) or parse_decimal_text(
        attr_of(soup.select_one("[data-areautil]"), "data-areautil")
    )
    detail["valor_m2"] = parse_decimal_text(text_of(soup.select_one("#valorM2Imovel")))
    detail["vagas"] = parse_detail_icon_value(soup, "vacancy")

    anunciante = attr_of(soup.select_one("#nomeDoAnunciante"), "value")
    if anunciante:
        detail["anunciante"] = anunciante
    creci = attr_of(soup.select_one("#creciDoAnunciante"), "value")
    if creci:
        detail["creci"] = creci

    endereco = attr_of(soup.select_one("#enderecoDoImovel"), "value")
    if endereco:
        detail["quadra"] = endereco.split(" - ")[-1].strip()
    else:
        detail["quadra"] = text_of(soup.select_one(".imovel-title h1"))

    return {key: value for key, value in detail.items() if value not in ("", None)}


def parse_detail_icon_value(soup: BeautifulSoup, class_name: str) -> str:
    element = soup.select_one(f".{class_name} span")
    match = re.search(r"\d+", text_of(element))
    return match.group(0) if match else ""


def regex_group(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def scrape(
    config: ScrapeConfig,
    arquivo_listagem: Path | list[Path] | None = None,
    checkpoint_path: Path | None = None,
    *,
    chrome: bool = False,
    resume: CollectionCheckpoint | None = None,
) -> list[dict[str, str]]:
    if chrome and not arquivo_listagem:
        try:
            from playwright.sync_api import Error, sync_playwright
        except ImportError as exc:
            raise ColetaIncompleta("Instale playwright para usar --chrome.") from exc
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chrome", headless=False)
                page = browser.new_page(locale="pt-BR")
                return scrape_pages(
                    config, arquivo_listagem, checkpoint_path,
                    partial(fetch_chrome, page, timeout=config.timeout),
                    resume=resume,
                )
        except Error as exc:
            raise ColetaIncompleta(f"Falha ao abrir ou executar o Chrome: {exc}") from exc

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        return scrape_pages(
            config, arquivo_listagem, checkpoint_path,
            partial(fetch, session, timeout=config.timeout,
                    retries=config.retries, backoff=config.backoff),
            resume=resume,
        )


def scrape_pages(
    config: ScrapeConfig,
    arquivo_listagem: Path | list[Path] | None,
    checkpoint_path: Path | None,
    download: Callable[[str], str],
    *,
    strict: bool = False,
    resume: CollectionCheckpoint | None = None,
) -> list[dict[str, str]]:

    ofertas = ["venda", "aluguel"] if config.oferta == "ambos" else [config.oferta]
    rows: list[dict[str, str]] = resume.rows() if resume else []
    seen_links: set[str] = {row["link"] for row in rows}
    if checkpoint_path and rows:
        write_csv(rows, checkpoint_path)
    if resume and (resume.complete or (config.limite and len(rows) >= config.limite)):
        return rows
    # (oferta, escopo, pagina, erro) — escopos que morreram apos esgotar as tentativas.
    escopos_com_falha: list[tuple[str, str, int, str]] = []

    def add_cards(cards: list[dict[str, str]]) -> bool:
        for row in cards:
            if row["link"] in seen_links:
                continue
            if config.detalhes and not arquivo_listagem:
                try:
                    print(f"Baixando detalhe: {row['link']}", file=sys.stderr)
                    detail = parse_detail(download(row["link"]))
                    if strict and not detail.get("id"):
                        raise ColetaIncompleta(f"Detalhe sem estrutura reconhecida: {row['link']}")
                    if strict and detail["id"] != row["id"]:
                        raise ColetaIncompleta(f"Detalhe pertence a outro anuncio: {row['link']}")
                    row.update(detail)
                    time.sleep(config.delay)
                except requests.RequestException as exc:
                    if resume:
                        raise ColetaIncompleta(f"Detalhe interrompido em {row['link']}: {exc}") from exc
                    print(f"Falha no detalhe {row['link']}: {exc}", file=sys.stderr)
            saved_row = {field: row.get(field, "") for field in CSV_FIELDS}
            if resume:
                resume.save_row(saved_row)
            rows.append(saved_row)
            seen_links.add(row["link"])
            if checkpoint_path:
                if len(rows) == 1:
                    write_csv(rows, checkpoint_path)
                else:
                    with checkpoint_path.open("a", newline="", encoding="utf-8") as file:
                        csv.DictWriter(file, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL).writerow(rows[-1])
            if config.limite and len(rows) >= config.limite:
                return True
        return False

    if arquivo_listagem:
        arquivos = [arquivo_listagem] if isinstance(arquivo_listagem, Path) else arquivo_listagem
        for arquivo in arquivos:
            try:
                raw = arquivo.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ColetaIncompleta(f"Nao foi possivel ler {arquivo}: {exc}") from exc
            cards = parse_listing(raw)
            print(f"{arquivo}: {len(cards)} imoveis encontrados", file=sys.stderr)
            if not cards:
                raise ColetaIncompleta(f"{arquivo} nao contem anuncios reconhecidos.")
            if add_cards(cards):
                break
        return rows

    tasks = [(oferta, scope) for oferta in ofertas for scope in build_scopes(config)]
    first_task, first_page = resume.position if resume else (0, config.inicio)
    for task_index in range(first_task, len(tasks)):
        oferta, scope = tasks[task_index]
        pagina = first_page if task_index == first_task else config.inicio
        seen_scope = resume.scope_links(task_index) if resume else set()
        while True:
            if config.fim is not None and pagina > config.fim:
                if resume:
                    resume.advance(task_index + 1, config.inicio)
                break
            url = build_listing_url(oferta, pagina, scope)
            print(f"Baixando listagem: {url}", file=sys.stderr)
            try:
                stored_page = resume.pending(task_index, pagina) if resume else None
                raw = stored_page if stored_page is not None else download(url)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    print(f"Parando {oferta}/{scope.label}: pagina {pagina} retornou 404.", file=sys.stderr)
                    if resume:
                        resume.advance(task_index + 1, config.inicio)
                    break
                if resume:
                    raise ColetaIncompleta(f"Listagem interrompida em {url}: {exc}") from exc
                escopos_com_falha.append((oferta, scope.label, pagina, str(exc)))
                print(f"FALHA {oferta}/{scope.label} pagina {pagina}: {exc}. Pulando escopo.", file=sys.stderr)
                break
            except requests.RequestException as exc:
                if resume:
                    raise ColetaIncompleta(f"Listagem interrompida em {url}: {exc}") from exc
                # Ja passou por todas as tentativas do fetch. Perder um escopo e
                # melhor do que perder a varredura inteira.
                escopos_com_falha.append((oferta, scope.label, pagina, str(exc)))
                print(f"FALHA {oferta}/{scope.label} pagina {pagina}: {exc}. Pulando escopo.", file=sys.stderr)
                break
            cards = parse_listing(raw)
            soup = make_soup(raw) if strict else None
            print(f"{url}: {len(cards)} imoveis encontrados", file=sys.stderr)
            if not cards:
                if strict:
                    heading = text_of(soup.select_one(".busca-middle-column h1"))
                    if not re.match(r"^0\s+Im[oó]ve(?:l|is)", heading, re.I):
                        raise ColetaIncompleta(f"Listagem sem estrutura reconhecida: {url}")
                print(f"Parando {oferta}/{scope.label}: pagina {pagina} sem imoveis.", file=sys.stderr)
                if resume:
                    resume.advance(task_index + 1, config.inicio)
                break
            if strict and stored_page is None and not ({card["link"] for card in cards} - seen_scope):
                raise ColetaIncompleta(f"O portal repetiu os anuncios em {url}")
            seen_scope.update(card["link"] for card in cards)
            if resume and stored_page is None:
                resume.save_page(task_index, pagina, raw, {card["link"] for card in cards})
            if add_cards(cards):
                if not escopos_com_falha:
                    return rows
                raise ColetaIncompleta("Limite atingido apos falha de listagem.")
            if strict:
                next_button = soup.select_one(".busca-pagination .next")
                heading = text_of(soup.select_one(".busca-middle-column h1"))
                total_match = re.match(r"^([\d.]+)\s+Im[oó]ve(?:l|is)", heading, re.I)
                total = int(total_match.group(1).replace(".", "")) if total_match else None
                finished = next_button and (
                    "disabled" in next_button.get("class", [])
                    or next_button.get("aria-disabled") == "true"
                )
                if not next_button:
                    if total is None or total > len(cards):
                        raise ColetaIncompleta(f"Paginacao ausente em {url}")
                    finished = True
                if finished:
                    if config.inicio == 1 and total is not None and len(seen_scope) < total:
                        raise ColetaIncompleta(f"{url}: {len(seen_scope)} anuncios coletados para {total} resultados")
                    if resume:
                        resume.advance(task_index + 1, config.inicio)
                    break
            pagina += 1
            if resume:
                resume.advance(task_index, pagina)
            time.sleep(config.delay)

    if escopos_com_falha:
        print("", file=sys.stderr)
        print("=== ESCOPOS INCOMPLETOS ===", file=sys.stderr)
        for oferta, label, pagina, erro in escopos_com_falha:
            print(f"  {oferta}/{label}: parou na pagina {pagina} ({erro})", file=sys.stderr)
        print(
            f"  {len(escopos_com_falha)} escopo(s) incompleto(s). "
            f"Rode de novo so eles com --cidades/--bairros e --inicio.",
            file=sys.stderr,
        )
        raise ColetaIncompleta("Uma ou mais listagens falharam.")

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scraper de imoveis do DF Imoveis.")
    parser.add_argument("--oferta", choices=["venda", "aluguel", "ambos"], default="ambos")
    parser.add_argument("--inicio", type=int, default=1, help="Pagina inicial.")
    parser.add_argument("--fim", type=int, help="Pagina final. Se omitido, continua ate uma pagina sem imoveis.")
    parser.add_argument("--delay", type=float, default=1.0, help="Pausa entre requisicoes, em segundos.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout HTTP, em segundos.")
    parser.add_argument("--retries", type=int, default=3,
                        help="Tentativas extras por requisicao que falhar (timeout, 5xx, 429).")
    parser.add_argument("--backoff", type=float, default=2.0,
                        help="Base da espera exponencial entre tentativas, em segundos.")
    parser.add_argument("--saida", help="Arquivo CSV de saida. Se omitido, usa a data da coleta.")
    parser.add_argument("--limite", type=int, help="Numero maximo de imoveis para capturar.")
    parser.add_argument("--sem-detalhes", action="store_true", help="Nao abrir paginas individuais dos imoveis.")
    parser.add_argument("--chrome", action="store_true", help="Coletar pelo Google Chrome visivel (requer playwright).")
    parser.add_argument("--chrome-nativo", action="store_true",
                        help="Receber HTML da extensao conectada ao Chrome normal.")
    parser.add_argument("--arquivo-listagem", type=Path, action="append",
                        help="HTML salvo no navegador. Repita a opcao para importar varias paginas.")
    parser.add_argument(
        "--escopo",
        choices=["bairros", "todos"],
        default="bairros",
        help="Modo bairros percorre cidade/bairro; todos usa a URL antiga /df/todos.",
    )
    parser.add_argument("--cidades", help="Lista separada por virgula. Ex: brasilia,aguas claras.")
    parser.add_argument("--bairros", help="Lista separada por virgula para usar nas cidades selecionadas.")
    return parser.parse_args()


def fetch_native(
    transport: Callable[[str], str], url: str, retries: int, backoff: float,
) -> str:
    for attempt in range(retries + 1):
        raw = transport(url)
        soup = make_soup(raw)
        title = soup.title.get_text().lower() if soup.title else ""
        challenge = (
            "just a moment" in title or "um momento" in title
            or "attention required" in title or soup.select_one("#challenge-form")
        )
        if not challenge:
            return raw
        if attempt == retries:
            raise PortalBloqueado(f"Verificacao persistente em {url}")
        wait = backoff ** (attempt + 1)
        print(f"Verificacao temporaria em {url}. Aguardando {wait:.0f}s antes de conferir novamente.",
              file=sys.stderr)
        time.sleep(wait)
    raise PortalBloqueado(f"Verificacao solicitada em {url}")


def main() -> int:
    args = parse_args()
    if args.inicio < 1 or (args.fim is not None and args.fim < args.inicio):
        raise SystemExit("--inicio deve ser >= 1 e --fim deve ser >= --inicio")
    if args.chrome_nativo and (args.chrome or args.arquivo_listagem):
        raise SystemExit("--chrome-nativo deve ser usado com coleta online e sem --chrome")
    if args.timeout <= 0 or args.delay < 0 or (args.limite is not None and args.limite <= 0):
        raise SystemExit("--timeout e --limite devem ser positivos; --delay deve ser >= 0")
    if args.retries < 0:
        raise SystemExit("--retries deve ser >= 0")

    config = ScrapeConfig(
        oferta=args.oferta,
        inicio=args.inicio,
        fim=args.fim,
        delay=args.delay,
        detalhes=not args.sem_detalhes,
        timeout=args.timeout,
        limite=args.limite,
        escopo=args.escopo,
        cidades=parse_csv_arg(args.cidades),
        bairros=parse_csv_arg(args.bairros),
        retries=args.retries,
        backoff=args.backoff,
    )
    output_path = Path(args.saida) if args.saida else Path(f"{date.today().isoformat()}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f"{output_path.stem}_", suffix=".part", delete=False
    ) as file:
        checkpoint = Path(file.name)
    resume = None
    try:
        with ExitStack() as resources:
            if not args.arquivo_listagem:
                ofertas = ["venda", "aluguel"] if config.oferta == "ambos" else [config.oferta]
                settings = {
                    "version": 1, "inicio": config.inicio, "fim": config.fim,
                    "limite": config.limite, "detalhes": config.detalhes,
                    "tasks": [build_listing_url(oferta, config.inicio, scope)
                              for oferta in ofertas for scope in build_scopes(config)],
                    "fields": CSV_FIELDS,
                }
                resume = resources.enter_context(CollectionCheckpoint(output_path, settings))
                task_index, page = resume.position
                position = "concluido" if resume.complete else f"busca {task_index + 1}, pagina {page}"
                print(f"Checkpoint: {resume.path} ({position})", file=sys.stderr)
            transport = None
            if resume and resume.complete:
                print("Coleta ja concluida neste checkpoint. Exportando os dados salvos.", file=sys.stderr)
                rows = resume.rows()
            elif args.chrome_nativo:
                transport = resources.enter_context(ChromeNativeTransport(config.timeout))
                print("Usando a extensao conectada ao Chrome normal.", file=sys.stderr)
                download_native = partial(fetch_native, transport,
                                          retries=config.retries, backoff=config.backoff)
                rows = scrape_pages(config, None, checkpoint, download_native, strict=True, resume=resume)
            else:
                rows = scrape(config, args.arquivo_listagem, checkpoint, chrome=args.chrome, resume=resume)
            if not rows:
                raise ColetaIncompleta("Nenhum anuncio encontrado na coleta.")
            write_csv(rows, checkpoint)
            checkpoint.replace(output_path)
            if resume:
                resume.mark_complete()
            if transport:
                transport.complete(len(rows), output_path)
    except KeyboardInterrupt:
        print("\nColeta interrompida. Repita o comando para continuar pelo checkpoint.", file=sys.stderr)
        return 130
    except (PortalBloqueado, ColetaIncompleta, NativeTransportError, CheckpointError,
            sqlite3.Error, OSError, json.JSONDecodeError) as exc:
        bloqueado = isinstance(exc, PortalBloqueado)
        titulo = "PORTAL BLOQUEADO" if bloqueado else "COLETA INCOMPLETA"
        print(f"\n=== {titulo} ===\n  {exc}", file=sys.stderr)
        return SAIDA_PORTAL_BLOQUEADO if bloqueado else 1
    finally:
        if checkpoint.exists():
            if checkpoint.stat().st_size:
                print(f"  Parcial guardado em {checkpoint}", file=sys.stderr)
            else:
                checkpoint.unlink()
    print(f"CSV gerado: {output_path} ({len(rows)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
