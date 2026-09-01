from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit
import requests
from bs4 import BeautifulSoup, Tag
BASE_URL = 'https://www.wimoveis.com.br'
CSV_FIELDS = ['id', 'link', 'codigo', 'creci', 'anunciante', 'tipo', 'tipo_imovel', 'area_util', 'bairro', 'cidade', 'preco', 'valor_m2', 'quartos', 'vagas', 'latitude', 'longitude', 'quadra', 'data']
BAIRROS_POR_CIDADE: dict[str, dict[str, str]] = {'brasilia': {'asa norte': 'asa-norte', 'asa sul': 'asa-sul', 'altiplano leste': 'altiplano-leste', 'granja do torto': 'granja-do-torto', 'jardim botanico': 'setor-habitacional-jardim-botanico', 'setor habitacional jardim botanico': 'setor-habitacional-jardim-botanico', 'jardins mangueiral': 'setor-habitacional-jardins-mangueiral', 'setor habitacional jardins mangueiral': 'setor-habitacional-jardins-mangueiral', 'lago norte': 'lago-norte', 'lago sul': 'lago-sul', 'noroeste': 'noroeste', 'octogonal': 'octogonal', 'park sul': 'park-sul', 'park way': 'park-way', 'setor industrial': 'zona-industrial', 'zona industrial': 'zona-industrial', 'setor tororo': 'setor-habitacional-tororo', 'setor habitacional tororo': 'setor-habitacional-tororo', 'sig': 'setor-de-industrias-graficas', 'sudoeste': 'sudoeste', 'taquari': 'taquari', 'vila da telebrasilia': 'vila-da-telebrasilia', 'vila planalto': 'vila-planalto', 'zona civico administrativa': 'zona-civico-administrativa', 'zona rural': 'zona-rural'}, 'aguas claras': {'aguas claras': 'aguas-claras', 'aguas claras norte': 'aguas-claras-norte', 'norte': 'aguas-claras-norte', 'aguas norte': 'aguas-claras-norte', 'aguas claras sul': 'aguas-claras-sul', 'sul': 'aguas-claras-sul', 'aguas sul': 'aguas-claras-sul', 'areal': 'areal', 'arniqueiras': 'arniqueiras', 'ade': 'ade'}}
CIDADES_SEM_BAIRRO = ['alphaville', 'brazlandia', 'candangolandia', 'ceilandia', 'cruzeiro', 'gama', 'guara', 'jardim botanico', 'nucleo bandeirante', 'paranoa', 'planaltina', 'riacho fundo', 'samambaia', 'santa maria', 'sao sebastiao', 'setor industrial', 'sobradinho', 'taguatinga', 'varjao', 'vicente pires', 'vila estrutural']
DF_CITY_KEYS = set(BAIRROS_POR_CIDADE) | set(CIDADES_SEM_BAIRRO)
CITY_SLUG_OVERRIDES = {'jardim botanico': 'jardim-botanico', 'nucleo bandeirante': 'nucleo-bandeirante', 'riacho fundo': 'riacho-fundo', 'sao sebastiao': 'sao-sebastiao', 'setor industrial': 'setor-industrial', 'vicente pires': 'vicente-pires', 'vila estrutural': 'vila-estrutural'}
STATUS_RETENTAVEIS = {403, 408, 425, 429, 500, 502, 503, 504}
PROPERTY_RE = re.compile('/propriedades/[^?#]+-(\\d+)\\.html', re.I)

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
    debug_html: bool = False
    modo_segmentado: bool = False
    max_segmentos: int | None = None

@dataclass(frozen=True)
class ListingSegment:
    categoria: str
    subtipo: str | None = None
    filtro: str | None = None

    @property
    def label(self) -> str:
        partes = [self.categoria]
        if self.subtipo:
            partes.append(self.subtipo)
        if self.filtro:
            partes.append(self.filtro)
        return '/'.join(partes)

@dataclass(frozen=True)
class ListingScope:
    cidade: str
    bairro: str | None = None
    bairro_slug: str | None = None

    @property
    def label(self) -> str:
        return f'{self.cidade}/{self.bairro}' if self.bairro else self.cidade

def normalize_space(value: str | None) -> str:
    if not value:
        return ''
    value = value.replace('\xa0', ' ')
    return re.sub('\\s+', ' ', value).strip()

def strip_accents(value: str) -> str:
    value = unicodedata.normalize('NFKD', value)
    return ''.join((char for char in value if not unicodedata.combining(char)))

def normalize_key(value: str | None) -> str:
    value = normalize_space(value).lower()
    return strip_accents(value)

def slugify(value: str) -> str:
    value = normalize_key(value)
    value = re.sub('[^a-z0-9]+', '-', value)
    return value.strip('-')

def only_digits(value: str | None) -> str:
    return re.sub('\\D+', '', value or '')

def parse_number(value: str | None) -> str:
    value = normalize_space(value)
    if not value:
        return ''
    match = re.search('\\d[\\d.\\s]*(?:,\\d+)?', value)
    if not match:
        return ''
    number = match.group(0).replace(' ', '')
    if ',' in number:
        integer_part = number.split(',', 1)[0]
        return re.sub('\\D+', '', integer_part)
    if number.count('.') == 1:
        left, right = number.split('.', 1)
        if 1 <= len(right) <= 2:
            return only_digits(left)
    return only_digits(number)

def parse_area_number(value: str | None) -> str:
    value = normalize_space(value)
    if not value:
        return ''
    match = re.search('\\d[\\d.\\s]*(?:,\\d+)?', value)
    if not match:
        return ''
    number = match.group(0).replace(' ', '')
    if ',' in number:
        left, right = number.rsplit(',', 1)
        if 1 <= len(right) <= 2:
            integer = re.sub('\\D+', '', left) or '0'
            return f'{integer}.{right}'.rstrip('0').rstrip('.')
        return only_digits(number)
    if number.count('.') == 1:
        left, right = number.split('.', 1)
        if 1 <= len(right) <= 2:
            integer = only_digits(left) or '0'
            return f'{integer}.{right}'.rstrip('0').rstrip('.')
    return only_digits(number)

def canonical_url(url: str) -> str:
    parts = urlsplit(urljoin(BASE_URL, url))
    return urlunsplit((parts.scheme or 'https', parts.netloc, parts.path, '', ''))

def text_of(element: Tag | None) -> str:
    return normalize_space(element.get_text(' ', strip=True)) if element else ''

def attr_of(element: Tag | None, name: str) -> str:
    if not element:
        return ''
    value = element.get(name)
    return normalize_space(str(value)) if value is not None else ''

def parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    values = [normalize_key(item) for item in value.split(',')]
    return [item for item in values if item]

def build_scopes(config: ScrapeConfig) -> list[ListingScope]:
    selected_cities = config.cidades or [*BAIRROS_POR_CIDADE, *CIDADES_SEM_BAIRRO]
    scopes: list[ListingScope] = []
    for cidade_raw in selected_cities:
        cidade = normalize_key(cidade_raw)
        if config.escopo == 'cidades':
            scopes.append(ListingScope(cidade=cidade))
            continue
        if cidade not in BAIRROS_POR_CIDADE:
            if cidade not in CIDADES_SEM_BAIRRO:
                valid = ', '.join([*BAIRROS_POR_CIDADE, *CIDADES_SEM_BAIRRO])
                raise SystemExit(f'Cidade sem mapeamento: {cidade_raw}. Cidades disponíveis: {valid}')
            if config.bairros:
                raise SystemExit(f'A cidade {cidade_raw} está configurada sem bairros. Remova --bairros ou use --escopo cidades.')
            scopes.append(ListingScope(cidade=cidade))
            continue
        mapping = BAIRROS_POR_CIDADE[cidade]
        selected_bairros = config.bairros or list(mapping)
        for bairro_raw in selected_bairros:
            bairro = normalize_key(bairro_raw)
            bairro_slug = mapping.get(bairro)
            if not bairro_slug:
                valid = ', '.join(mapping)
                raise SystemExit(f'Bairro sem mapeamento para {cidade}: {bairro_raw}. Bairros disponíveis: {valid}')
            scopes.append(ListingScope(cidade=cidade, bairro=bairro, bairro_slug=bairro_slug))
    unique: dict[tuple[str, str | None], ListingScope] = {}
    for scope in scopes:
        unique[scope.cidade, scope.bairro_slug] = scope
    return list(unique.values())

def city_slug(city: str) -> str:
    return CITY_SLUG_OVERRIDES.get(city, slugify(city))

def build_listing_url(oferta: str, scope: ListingScope) -> str:
    if scope.cidade not in DF_CITY_KEYS:
        raise SystemExit(f'Cidade fora do Distrito Federal ou sem mapeamento: {scope.cidade}')
    base = f'{BASE_URL}/{oferta}/imoveis/df/{city_slug(scope.cidade)}'
    if scope.bairro_slug:
        return f'{base}/{scope.bairro_slug}'
    return base
SEGMENT_ORDERS = [None, 'ordem-precio-menor']
ROOM_FILTERS = ['1-quarto', '2-quartos', '3-quartos', '4-quartos', 'mais-de-4-quartos']
APARTMENT_SUBTYPES = ['padrao', 'flat', 'cobertura', 'duplex']
COMMERCIAL_SUBTYPES = ['ponto-comercial', 'loja-de-shopping-centro-comercial', 'casa-comercial']
COMMERCIAL_KEYWORDS = ['q-sala', 'q-loja', 'q-predio', 'q-galpao', 'q-clinica']
APARTMENT_DISCOVERY_FILTERS = ['areac-elevador', 'areap-varanda', 'areap-mobiliado', 'areac-proximo-ao-metro', 'areap-varanda-gourmet', 'areac-churrasqueira', 'areac-piscina-aquecida', 'antiquity-breve-lancamento']
HOUSE_DISCOVERY_FILTERS = ['areac-piscina', 'areap-varanda']
COMMERCIAL_DISCOVERY_FILTERS = ['areac-elevador']
SEGMENT_REPORT_FIELDS = ['oferta', 'escopo', 'segmento', 'total_informado', 'cards_unicos', 'status', 'novos_globais', 'total_global_acumulado', 'urls_consultadas']

def build_segment_url(oferta: str, scope: ListingScope, segment: ListingSegment, order: str | None=None) -> str:
    if scope.cidade not in DF_CITY_KEYS:
        raise SystemExit(f'Cidade fora do Distrito Federal ou sem mapeamento: {scope.cidade}')
    parts = [BASE_URL, oferta, segment.categoria]
    if segment.subtipo:
        parts.append(segment.subtipo)
    parts.extend(['df', city_slug(scope.cidade)])
    if scope.bairro_slug:
        parts.append(scope.bairro_slug)
    if segment.filtro:
        parts.append(segment.filtro)
    if order:
        parts.append(order)
    return '/'.join((part.strip('/') for part in parts if part))

def extract_listing_total(raw: str) -> int | None:
    soup = BeautifulSoup(raw, 'html.parser')
    h1 = soup.find('h1')
    if not h1:
        return None
    text = text_of(h1)
    match = re.match('\\s*([\\d.]+)\\b', text)
    if not match:
        return None
    try:
        return int(match.group(1).replace('.', ''))
    except ValueError:
        return None

def write_segment_report(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=SEGMENT_REPORT_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8', 'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7', 'Cache-Control': 'no-cache', 'Pragma': 'no-cache', 'Upgrade-Insecure-Requests': '1'})
    return session

def fetch(session: requests.Session, url: str, timeout: int, retries: int=3, backoff: float=2.0) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            if response.status_code in STATUS_RETENTAVEIS:
                raise requests.HTTPError(f'status {response.status_code}', response=response)
            response.raise_for_status()
            if not response.encoding or response.encoding.lower() == 'iso-8859-1':
                response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
        except requests.RequestException as exc:
            last_error = exc
        if attempt < retries:
            wait = backoff ** attempt
            status = ''
            if isinstance(last_error, requests.HTTPError) and last_error.response is not None:
                status = f' HTTP {last_error.response.status_code}'
            print(f'  falha{status} ({type(last_error).__name__}); tentativa {attempt + 2}/{retries + 1} em {wait:.0f}s', file=sys.stderr)
            time.sleep(wait)
    assert last_error is not None
    raise last_error

def extract_jsonld(soup: BeautifulSoup) -> list[Any]:
    result: list[Any] = []
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.string or script.get_text()
        raw = raw.strip() if raw else ''
        if not raw:
            continue
        try:
            result.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return result

def walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)

def find_json_value(objects: Iterable[Any], keys: set[str]) -> Any:
    for obj in objects:
        for node in walk_json(obj):
            for key in keys:
                if key in node and node[key] not in (None, '', [], {}):
                    return node[key]
    return None

def property_id_from_url(url: str) -> str:
    match = PROPERTY_RE.search(url)
    return match.group(1) if match else ''

def listing_property_links(soup: BeautifulSoup) -> list[Tag]:
    seen: set[str] = set()
    links: list[Tag] = []
    for anchor in soup.find_all('a', href=True):
        href = attr_of(anchor, 'href')
        if not PROPERTY_RE.search(href):
            continue
        link = canonical_url(href)
        if link in seen:
            continue
        seen.add(link)
        links.append(anchor)
    return links

def find_card_container(anchor: Tag) -> Tag:
    fallback: Tag = anchor
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name in {'body', 'html'}:
            break
        text = text_of(parent)
        if len(text) > 4000:
            break
        fallback = parent
        has_price = 'R$' in text or 'Consultar preço' in text
        has_feature = bool(re.search('\\bm\\s*[²2]\\b|\\bquartos?\\b|\\bvagas?\\b|\\bban(?:heiro)?s?\\b', text, re.I))
        own_property_links = {canonical_url(attr_of(a, 'href')) for a in parent.find_all('a', href=True) if PROPERTY_RE.search(attr_of(a, 'href'))}
        if has_price and has_feature and (len(own_property_links) == 1):
            return parent
    return fallback

def detect_tipo_imovel(text: str, primary_type: str='', title: str='') -> str:
    key = normalize_key(text)
    primary = normalize_key(primary_type)
    title_key = normalize_key(title)
    primary_map = {'apartamento': 'apartamento', 'casa': 'casa', 'terreno': 'lote', 'lote': 'lote', 'sala': 'sala', 'sala comercial': 'sala', 'cobertura': 'cobertura', 'galpao': 'galpao', 'predio': 'predio', 'flat': 'apartamento', 'kitnet': 'apartamento', 'studio': 'apartamento', 'garagem': 'garagem'}
    garage_title = bool(re.search('^(?:vagas? de garagem\\b|box(?: de)? garagem\\b|garagem(?:\\s+(?:a venda|coberta|no|na|e depositos))?\\b|\\d+ vagas? de garagem\\b)', title_key) or re.search('\\bvaga de garagem a venda\\b', title_key))
    if garage_title:
        return 'garagem'
    if re.search('^casa\\b|\\bcasa na asa sul\\b|\\bcasa na asa norte\\b', title_key) and (not re.search('^casa loterica\\b|^casa comercial\\b', title_key)):
        return 'casa'
    if re.search('^terreno\\b|\\blote comercial\\b|\\blote beira lago\\b', title_key) and (not re.search('\\bpredio\\b|\\bedificio\\b', title_key)):
        return 'lote'
    if primary == 'apartamento' and re.search('\\bcobertura\\b', title_key):
        return 'cobertura'
    if primary in primary_map:
        return primary_map[primary]
    if primary == 'comercial':
        if re.search('\\bvaga de garagem a venda\\b|\\bvendo excelente vaga de garagem\\b|\\bvendemos.{0,80}vaga de garagem\\b', key):
            return 'garagem'
        if re.search('\\bapartamento\\b|\\bflat\\b|\\bhotel[- ]?flat\\b|\\bkitnet\\b|\\bkitinete\\b', title_key):
            return 'apartamento'
        strong_building_title = bool(re.search('(?:^|[-–—:]\\s)(?:excelente\\s+)?predio(?:\\s+(?:inteiro|comercial))?\\b|\\bpredio\\s+inteiro\\b|\\bedificio\\s+inteiro\\b|\\bedificio\\s+comercial\\s+inteiro\\b', title_key))
        if strong_building_title:
            return 'predio'
        if re.search('\\blojas?\\b|\\bponto comercial\\b', title_key):
            return 'ponto comercial'
        if re.search('\\bsala\\b|\\bconsultorios?\\b|\\bclinicas?\\b', title_key):
            return 'sala'
        if re.search('\\bgalpao\\b|\\bdeposito\\b', title_key):
            return 'galpao'
        if re.search('\\blojas?\\b|\\bponto comercial\\b', key):
            return 'ponto comercial'
        if re.search('\\bconsultorios?\\b|\\bclinicas?\\b|\\bsalas?\\s+comerciais?\\b|\\bconjuntos?\\s+comerciais?\\b|\\bescritorios?\\b|\\bsala\\b', key):
            return 'sala'
        if re.search('\\bgalpao\\b|\\bdeposito\\b', key):
            return 'galpao'
        if re.search('\\bpredio\\s+(?:inteiro|comercial)\\b|\\bedificio\\s+inteiro\\b', key):
            return 'predio'
        return 'comercial'
    residential_signal = bool(re.search('\\bapartamento\\b|\\b[2-9]\\s+quartos?\\b|\\bsqs\\b|\\bshigs\\b', title_key))
    if re.search('\\bcoberturas?\\b', title_key):
        return 'cobertura'
    if re.search('\\bapartamento\\b|\\bflat\\b|\\bkitnet\\b|\\bkitinete\\b|\\bkit\\b|\\bstudio\\b', title_key):
        return 'apartamento'
    if re.search('\\bcasa\\b|\\bsobrado\\b', title_key):
        return 'casa'
    if re.search('\\bterreno\\b|\\bloteamento\\b|\\blote\\b', title_key):
        return 'lote'
    if re.search('\\bgalpao\\b|\\bdeposito\\b', title_key):
        return 'galpao'
    if re.search('\\bpredio\\s+(?:inteiro|comercial)\\b|\\bedificio\\s+inteiro\\b', title_key):
        return 'predio'
    if re.search('\\blojas?\\b|\\bponto comercial\\b', title_key):
        return 'ponto comercial'
    if not residential_signal and re.search('\\bconsultorios?\\b|\\bclinicas?\\b|\\bsalas?\\s+comerciais?\\b|\\bconjuntos?\\s+comerciais?\\b|\\bescritorios?\\b|\\bsala\\b', title_key):
        return 'sala'
    if re.search('\\bapartamento\\b|\\bflat\\b|\\bkitnet\\b|\\bkitinete\\b|\\bkit\\b|\\bstudio\\b', key):
        return 'apartamento'
    if re.search('\\bcasa\\b|\\bsobrado\\b', key):
        return 'casa'
    if re.search('\\bcoberturas?\\b', key):
        return 'cobertura'
    if re.search('\\bterreno\\b|\\bloteamento\\b|\\blote\\b', key):
        return 'lote'
    if re.search('\\bgalpao\\b|\\bdeposito\\b', key):
        return 'galpao'
    if re.search('\\blojas?\\b|\\bponto comercial\\b', key):
        return 'ponto comercial'
    if re.search('\\bpredio\\s+(?:inteiro|comercial)\\b|\\bedificio\\s+inteiro\\b', key):
        return 'predio'
    if re.search('\\bcomercial\\b', key):
        return 'comercial'
    if re.search('\\bshigs\\b', title_key):
        return 'casa'
    if re.search('\\b(?:sqs|crs|scrs|scs)\\b', title_key) and re.search('\\b\\d+\\s+quartos?\\b', title_key):
        return 'apartamento'
    if re.search('\\b\\d+\\s+quartos?\\b', title_key) and (not re.search('\\bloja\\b|\\bsala\\b|\\bpredio\\b|\\bgalpao\\b|\\bterreno\\b|\\blote\\b', title_key)):
        return 'apartamento'
    return 'imovel'

def first_regex(text: str, patterns: Iterable[str], flags: int=re.I) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return normalize_space(match.group(1))
    return ''

def parse_location_text(value: str) -> tuple[str, str, str]:
    value = normalize_space(value)
    if not value:
        return ('', '', '')
    parts = [normalize_space(part) for part in value.split(',') if normalize_space(part)]
    if len(parts) >= 3:
        cidade = parts[-1]
        bairro = parts[-2]
        quadra = ', '.join(parts[:-2])
        return (bairro.upper(), cidade.upper(), quadra)
    return ('', '', value)

def extract_card_location(container: Tag) -> tuple[str, str, str]:
    selectors = ['[data-qa*="LOCATION"]', '[data-qa*="ADDRESS"]', '[class*="location"]', '[class*="address"]', 'h2', 'h3', 'h4']
    candidates: list[str] = []
    for selector in selectors:
        for element in container.select(selector):
            text = text_of(element)
            if text and 'R$' not in text and (text not in candidates):
                candidates.append(text)
    for candidate in candidates:
        bairro, cidade, quadra = parse_location_text(candidate)
        if bairro and cidade:
            return (bairro, cidade, quadra)
    for candidate in candidates:
        if len(candidate) <= 180 and (not re.search('\\bquartos?\\b|\\bvagas?\\b|\\bm²\\b', candidate, re.I)):
            return ('', '', candidate)
    return ('', '', '')

def parse_card(anchor: Tag, oferta: str, scope: ListingScope | None) -> dict[str, str]:
    container = find_card_container(anchor)
    text = text_of(container)
    link = canonical_url(attr_of(anchor, 'href'))
    listing_id = property_id_from_url(link)
    preco = parse_number(first_regex(text, ['(?:^|\\s)R\\$\\s*([\\d.]+(?:,\\d+)?)', 'a partir de\\s+R\\$\\s*([\\d.]+(?:,\\d+)?)']))
    area = parse_number(first_regex(text, ['([\\d.,]+)\\s*m\\s*[²2]\\s*(?:tot\\.?|total)', '([\\d.,]+)\\s*m\\s*[²2]\\s*(?:útil|util)', '([\\d.,]+)\\s*m\\s*[²2]']))
    quartos = only_digits(first_regex(text, ['(\\d+)\\s*quartos?']))
    vagas = only_digits(first_regex(text, ['(\\d+)\\s*vagas?']))
    bairro, cidade, quadra = extract_card_location(container)
    if not cidade and scope:
        cidade = scope.cidade.upper()
    if not bairro and scope and scope.bairro:
        bairro = scope.bairro.upper()
    tipo_source = text
    image = container.find('img', alt=True)
    if image:
        tipo_source = f"{attr_of(image, 'alt')} {tipo_source}"
    return {'id': listing_id, 'link': link, 'codigo': '', 'creci': '', 'anunciante': '', 'tipo': oferta, 'tipo_imovel': detect_tipo_imovel(tipo_source), 'area_util': area, 'bairro': bairro, 'cidade': cidade, 'preco': preco, 'valor_m2': '', 'quartos': quartos, 'vagas': vagas, 'latitude': '', 'longitude': '', 'quadra': quadra, 'data': date.today().isoformat()}

def parse_listing(raw: str, oferta: str, scope: ListingScope | None=None, current_url: str | None=None, current_page: int=1) -> tuple[list[dict[str, str]], str | None]:
    soup = BeautifulSoup(raw, 'html.parser')
    cards = [parse_card(anchor, oferta, scope) for anchor in listing_property_links(soup)]
    return (cards, None)

def regex_group(text: str, patterns: Iterable[str], flags: int=re.I | re.S) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = match.group(1) if match.lastindex else match.group(0)
            return normalize_space(value)
    return ''
BRAZILIAN_UF_ALIASES = {'ac': 'AC', 'acre': 'AC', 'al': 'AL', 'alagoas': 'AL', 'ap': 'AP', 'amapa': 'AP', 'am': 'AM', 'amazonas': 'AM', 'ba': 'BA', 'bahia': 'BA', 'ce': 'CE', 'ceara': 'CE', 'df': 'DF', 'distrito federal': 'DF', 'es': 'ES', 'espirito santo': 'ES', 'go': 'GO', 'goias': 'GO', 'ma': 'MA', 'maranhao': 'MA', 'mt': 'MT', 'mato grosso': 'MT', 'ms': 'MS', 'mato grosso do sul': 'MS', 'mg': 'MG', 'minas gerais': 'MG', 'pa': 'PA', 'para': 'PA', 'pb': 'PB', 'paraiba': 'PB', 'pr': 'PR', 'parana': 'PR', 'pe': 'PE', 'pernambuco': 'PE', 'pi': 'PI', 'piaui': 'PI', 'rj': 'RJ', 'rio de janeiro': 'RJ', 'rn': 'RN', 'rio grande do norte': 'RN', 'rs': 'RS', 'rio grande do sul': 'RS', 'ro': 'RO', 'rondonia': 'RO', 'rr': 'RR', 'roraima': 'RR', 'sc': 'SC', 'santa catarina': 'SC', 'sp': 'SP', 'sao paulo': 'SP', 'se': 'SE', 'sergipe': 'SE', 'to': 'TO', 'tocantins': 'TO'}

def normalize_brazilian_uf(value: str | None) -> str:
    key = normalize_key(value)
    return BRAZILIAN_UF_ALIASES.get(key, '')

def parse_detail(raw: str, link: str='') -> dict[str, str]:
    soup = BeautifulSoup(raw, 'html.parser')
    text = text_of(soup)
    jsonld = extract_jsonld(soup)
    detail: dict[str, str] = {}
    listing_id = property_id_from_url(link)
    if listing_id:
        detail['id'] = listing_id
    operation = regex_group(text, ['\\b(venda|aluguel)\\s+R\\$', '\\b(venda|aluguel)\\b'])
    if operation:
        detail['tipo'] = normalize_key(operation)
    title = ''
    h1 = soup.find('h1')
    if h1:
        title = text_of(h1)
    if not title:
        name_json = find_json_value(jsonld, {'name', 'headline'})
        if isinstance(name_json, str):
            title = normalize_space(name_json)
    primary_type = regex_group(text[:1800], ['\\b(Apartamento|Casa|Comercial|Terreno|Lote|Sala(?: Comercial)?|Cobertura|Galp[aã]o|Pr[eé]dio|Flat|Kitnet|Studio|Garagem)\\s*[·•]'])
    if normalize_key(primary_type) == 'comercial':
        type_source = ' '.join((part for part in [title, text[:1800]] if part))
    else:
        type_source = title or text[:500]
    if type_source or primary_type:
        detected_type = detect_tipo_imovel(type_source, primary_type, title)
        if detected_type == 'imovel' and link:
            slug_text = urlsplit(link).path.rsplit('/', 1)[-1].replace('-', ' ')
            slug_detected = detect_tipo_imovel(slug_text, '', slug_text)
            if slug_detected != 'imovel':
                detected_type = slug_detected
        detail['tipo_imovel'] = detected_type
    offers = find_json_value(jsonld, {'offers'})
    if isinstance(offers, dict):
        price = offers.get('price') or offers.get('lowPrice')
        if price not in (None, ''):
            detail['preco'] = parse_number(str(price))
    if 'preco' not in detail:
        price_text = regex_group(text, ['\\b(?:venda|aluguel)\\s+R\\$\\s*([\\d.]+(?:,\\d+)?)', '\\bR\\$\\s*([\\d.]+(?:,\\d+)?)'])
        if price_text:
            detail['preco'] = parse_number(price_text)
    area_range = re.search('[\\d.,]+\\s*a\\s*[\\d.,]+\\s*m\\s*[²2]', text, re.I)
    if area_range:
        detail['_area_range'] = '1'
    else:
        area_text = regex_group(text, ['([\\d.,]+)\\s*m\\s*[²2]\\s*[uú]til', '(?:Apartamento|Casa|Comercial|Terreno|Sala|Cobertura)\\s*[·-]\\s*([\\d.,]+)\\s*m\\s*[²2]', '([\\d.,]+)\\s*m\\s*[²2]\\s*tot'])
        if area_text:
            area_value = parse_area_number(area_text)
            try:
                absurd_area = float(area_value or 0) > 1000
            except ValueError:
                absurd_area = False
            if absurd_area:
                decimal_candidates = re.findall('(?:area(?:\\s+util|\\s+privativa)?\\s*[:=-]?\\s*)?(\\d{1,3}[,.]\\d{1,2})\\s*m\\s*[²2]', text, re.I)
                for candidate in decimal_candidates:
                    parsed_candidate = parse_area_number(candidate)
                    try:
                        numeric_candidate = float(parsed_candidate)
                    except (TypeError, ValueError):
                        continue
                    if 5 <= numeric_candidate <= 1000:
                        area_value = parsed_candidate
                        break
            detail['area_util'] = area_value
    quartos_range = re.search('\\b\\d+\\s*a\\s*\\d+\\s*(?:quartos?|dormit[oó]rios?)\\b', text, re.I)
    if quartos_range:
        detail['_quartos_range'] = '1'
    else:
        quartos = regex_group(text, ['\\b(\\d+)\\s*quartos?\\b', '\\b(\\d+)\\s*dormit[oó]rios?\\b'])
        if quartos:
            detail['quartos'] = only_digits(quartos)
    vagas = regex_group(text, ['\\b(\\d+)\\s*vagas?\\b'])
    if vagas:
        detail['vagas'] = only_digits(vagas)
    if re.search('\\b(?:venda|aluguel)\\s+Consultar pre[cç]o\\b', text, re.I):
        detail['_price_consult'] = '1'
    address_text = ''
    for heading in soup.find_all(['h2', 'h3', 'h4', 'h5']):
        candidate = text_of(heading)
        if candidate.count(',') >= 2 and 'R$' not in candidate:
            address_text = candidate
            break
    if not address_text:
        address = find_json_value(jsonld, {'address'})
        if isinstance(address, dict):
            street = normalize_space(str(address.get('streetAddress') or ''))
            district = normalize_space(str(address.get('addressLocality') or ''))
            city = normalize_space(str(address.get('addressCity') or ''))
            region = normalize_space(str(address.get('addressRegion') or ''))
            uf = normalize_brazilian_uf(region)
            if uf:
                detail['_uf'] = uf
            address_text = ', '.join([part for part in [street, district, city] if part])
        elif isinstance(address, str):
            address_text = normalize_space(address)
    if '_uf' not in detail:
        region = regex_group(raw, ['["\\\']addressRegion["\\\']\\s*:\\s*["\\\']([^"\\\']+)', '\\bDistrito Federal\\b'])
        uf = normalize_brazilian_uf(region)
        if uf:
            detail['_uf'] = uf
    if address_text:
        bairro, cidade, quadra = parse_location_text(address_text)
        if bairro:
            detail['bairro'] = bairro
        if cidade:
            detail['cidade'] = cidade
        if quadra:
            detail['quadra'] = quadra
    geo = find_json_value(jsonld, {'geo'})
    if isinstance(geo, dict):
        lat = geo.get('latitude')
        lon = geo.get('longitude')
        if lat not in (None, ''):
            detail['latitude'] = str(lat)
        if lon not in (None, ''):
            detail['longitude'] = str(lon)
    if 'latitude' not in detail:
        lat = regex_group(raw, ['["\\\']latitude["\\\']\\s*:\\s*["\\\']?(-?\\d+(?:\\.\\d+)?)', '\\blatitude\\s*=\\s*["\\\']?(-?\\d+(?:\\.\\d+)?)'])
        if lat:
            detail['latitude'] = lat
    if 'longitude' not in detail:
        lon = regex_group(raw, ['["\\\']longitude["\\\']\\s*:\\s*["\\\']?(-?\\d+(?:\\.\\d+)?)', '\\blongitude\\s*=\\s*["\\\']?(-?\\d+(?:\\.\\d+)?)'])
        if lon:
            detail['longitude'] = lon
    creci = regex_group(text, ['CRECI\\s*[-/]?\\s*DF\\s*:?\\s*(\\d[\\d.]*\\s*CJ)\\b', 'CRECI\\s*[-/]?\\s*DF\\s*:?\\s*(\\d[\\d.]*)\\b', 'CRECI\\s*[-–—/]?\\s*(?:J|PJ|F)?\\s*:?\\s*(\\d[\\d.]*)(?:\\s*/\\s*DF)?\\b', '\\bCJ\\s*[:#-]?\\s*(\\d[\\d.]*)\\b'])
    if creci:
        detail['creci'] = creci
    codigo = regex_group(text, ['\\bChave\\s+do\\s+an[uú]ncio\\s*:\\s*([A-Za-z0-9][A-Za-z0-9._/-]*)', '\\bC[oó]digo\\s+do\\s+An[uú]ncio\\s*:\\s*([A-Za-z0-9][A-Za-z0-9._/-]*)', '\\bC[oó]digo\\s+do\\s+Im[oó]vel\\s*:\\s*([A-Za-z0-9][A-Za-z0-9._/-]*)', '\\bC[oó]digo\\s*:\\s*([A-Za-z0-9][A-Za-z0-9._/-]*)', '\\bRef(?:er[eê]ncia)?\\.?\\s*:\\s*([A-Za-z0-9][A-Za-z0-9._/-]*)', '\\bC[oó]d\\.?\\s*:\\s*([A-Za-z0-9][A-Za-z0-9._/-]*)'])
    if codigo:
        codigo = codigo.rstrip('._-/')
        if codigo:
            detail['codigo'] = codigo
    seller = find_json_value(jsonld, {'seller'})
    if isinstance(seller, dict) and seller.get('name'):
        detail['anunciante'] = normalize_space(str(seller['name']))
    if 'anunciante' not in detail:
        for selector in ['[data-qa*="PUBLISHER"]', '[class*="publisher"]', '[class*="advertiser"]', '[class*="seller"]']:
            element = soup.select_one(selector)
            candidate = text_of(element)
            if candidate and len(candidate) <= 120:
                detail['anunciante'] = candidate
                break
    if 'anunciante' not in detail and soup.title:
        page_title = text_of(soup.title)
        publisher = regex_group(page_title, ['publicado por\\s+(.+?)\\s+-\\s+Wimoveis\\b'])
        if publisher:
            detail['anunciante'] = publisher
    if 'anunciante' not in detail:
        company_creci = regex_group(text, ["([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ0-9 &.'-]{1,90}\\s+(?:Im[oó]veis|Imobili[aá]ria|Neg[oó]cios\\s+Imobili[aá]rios|Assessoria\\s+Imobili[aá]ria))\\s*[-–—]*\\s*(?:CRECI|CJ)\\b"])
        if company_creci:
            detail['anunciante'] = company_creci
    if 'anunciante' not in detail:
        company = regex_group(text, ["\\b(?:A|O)\\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ0-9 &.'-]{1,75}\\s+(?:Im[oó]veis|Imobili[aá]ria|Neg[oó]cios\\s+Imobili[aá]rios|Assessoria\\s+Imobili[aá]ria))\\s+(?:vende|apresenta|oferece)\\b", "(?:^|[.!?]\\s)([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ0-9 &.'-]{1,75}\\s+(?:Im[oó]veis|Imobili[aá]ria|Neg[oó]cios\\s+Imobili[aá]rios|Assessoria\\s+Imobili[aá]ria))\\s+(?:vende|apresenta|oferece)\\b", "\\bFale\\s+com\\s+a\\s+equipe\\s+da\\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ0-9 &.'-]{1,75}\\s+(?:Im[oó]veis|Imobili[aá]ria))\\b"])
        if company:
            detail['anunciante'] = company
    return {key: value for key, value in detail.items() if value not in (None, '')}

def normalize_output_row(row: dict[str, str]) -> None:
    if normalize_key(row.get('cidade')) == 'brasilia':
        row['cidade'] = 'BRASÍLIA'
    if normalize_key(row.get('bairro')) in {'2 asa sul', 'asa sul 2'}:
        row['bairro'] = 'ASA SUL'
    non_residential = {'sala', 'ponto comercial', 'comercial', 'predio', 'lote', 'galpao', 'garagem'}
    if normalize_key(row.get('tipo_imovel')) in non_residential:
        row['quartos'] = ''

def calculate_valor_m2(row: dict[str, str]) -> None:
    try:
        preco = int(float(row.get('preco') or 0))
        area = float(row.get('area_util') or 0)
    except (TypeError, ValueError):
        return
    if preco > 0 and area > 0:
        row['valor_m2'] = str(round(preco / area))
    else:
        row['valor_m2'] = ''

def row_is_df(row: dict[str, str]) -> bool:
    uf_raw = row.get('_uf')
    uf = normalize_brazilian_uf(uf_raw)
    if uf and uf != 'DF':
        return False
    return True

def scope_matches_row(scope: ListingScope, row: dict[str, str]) -> bool:
    row_city = normalize_key(row.get('cidade'))
    row_bairro = normalize_key(row.get('bairro'))
    if row_city and normalize_key(scope.cidade) not in row_city and (row_city not in normalize_key(scope.cidade)):
        return False
    if scope.bairro and row_bairro:
        wanted = normalize_key(scope.bairro)
        if wanted == 'asa sul':
            quadra_key = normalize_key(row.get('quadra'))
            link_key = normalize_key(row.get('link'))
            strong_other = bool(re.search('\\bsqn\\b|\\bsqsw\\b|\\bsqnw\\b|\\bsgcv\\b|\\bsmas\\b|\\bsig(?: sul)?\\b|\\bsetor sudoeste\\b|\\brodovia df 097\\b', quadra_key) or re.search('\\btaguatinga\\b|\\bnoroeste\\b|\\bsudoeste\\b|\\basa norte\\b', link_key))
            if not strong_other and 'park sul' in link_key:
                if not re.search('\\bsgas 910\\b|\\bw4 sul 910\\b|\\bmix park sul\\b', quadra_key):
                    strong_other = True
            if strong_other:
                return False
        if wanted == 'asa norte':
            quadra_key = normalize_key(row.get('quadra'))
            link_key = normalize_key(row.get('link'))
            strong_other = bool(re.search('^(?:sqnw\\b|sqsw\\b|sqs\\b|saus\\b|sces\\b|seps\\b|shin\\s+qi\\b)', quadra_key) or re.search('^ql\\s*\\d+.*lago norte\\b|^qi\\s*\\d+.*lago norte\\b', quadra_key) or re.search('^setor habitacional jardim botanico\\b', quadra_key) or re.search('\\bsqnw[- ]|\\blago-norte\\b|\\bjardim-botanico\\b|\\basa-sul\\b|\\bsudoeste\\b|\\bpark-sul\\b', link_key))
            if strong_other:
                return False
        aliases = {'jardim botanico': 'setor habitacional jardim botanico', 'jardins mangueiral': 'setor habitacional jardins mangueiral', 'setor tororo': 'setor habitacional tororo', 'norte': 'aguas norte', 'aguas claras norte': 'aguas norte', 'sul': 'aguas sul', 'aguas claras sul': 'aguas sul'}
        wanted_alt = aliases.get(wanted, wanted)
        row_alt = aliases.get(row_bairro, row_bairro)
        if wanted not in row_bairro and row_bairro not in wanted and (wanted_alt not in row_alt) and (row_alt not in wanted_alt):
            return False
    return True

def save_debug_html(raw: str, prefix: str, page: int) -> None:
    folder = Path('debug_html')
    folder.mkdir(exist_ok=True)
    filename = f'{slugify(prefix)}-pagina-{page}.html'
    (folder / filename).write_text(raw, encoding='utf-8', errors='replace')

def _merge_card(target: dict[str, dict[str, str]], row: dict[str, str]) -> None:
    link = row.get('link', '')
    if not link:
        return
    if link not in target:
        target[link] = row
        return
    current = target[link]
    for key, value in row.items():
        if value and (not current.get(key)):
            current[key] = value

def _fetch_segment_leaf(session: requests.Session, config: ScrapeConfig, oferta: str, scope: ListingScope, segment: ListingSegment, segment_counter: list[int]) -> tuple[dict[str, dict[str, str]], int | None, list[str]]:
    cards_by_link: dict[str, dict[str, str]] = {}
    urls: list[str] = []
    total: int | None = None
    for idx, order in enumerate(SEGMENT_ORDERS):
        if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
            break
        if idx > 0 and total is not None and (total <= 30):
            break
        url = build_segment_url(oferta, scope, segment, order)
        urls.append(url)
        segment_counter[0] += 1
        print(f'Segmento {segment_counter[0]}: {oferta}/{scope.label} | {segment.label}' + (f' | {order}' if order else ' | relevancia'), file=sys.stderr)
        print(f'  {url}', file=sys.stderr)
        try:
            raw = fetch(session, url, config.timeout, config.retries, config.backoff)
        except requests.RequestException as exc:
            print(f'  falha na listagem segmentada: {exc}', file=sys.stderr)
            continue
        if config.debug_html:
            suffix = slugify(f"{segment.label}-{order or 'relevancia'}")
            save_debug_html(raw, f'segmentado-{oferta}-{scope.label}-{suffix}', 1)
        if total is None:
            total = extract_listing_total(raw)
        cards, _ = parse_listing(raw, oferta, scope, url, 1)
        for row in cards:
            _merge_card(cards_by_link, row)
        print(f"  total informado={(total if total is not None else '?')} | cards desta URL={len(cards)} | unicos acumulados={len(cards_by_link)}", file=sys.stderr)
        time.sleep(config.delay)
    return (cards_by_link, total, urls)

def scrape_segmented(config: ScrapeConfig, checkpoint_path: Path | None=None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    session = make_session()
    ofertas = ['venda', 'aluguel'] if config.oferta == 'ambos' else [config.oferta]
    all_cards: dict[str, dict[str, str]] = {}
    report: list[dict[str, str]] = []
    segment_counter = [0]

    def harvest(oferta: str, scope: ListingScope, segment: ListingSegment) -> tuple[int | None, int]:
        cards, total, urls = _fetch_segment_leaf(session, config, oferta, scope, segment, segment_counter)
        before_global = len(all_cards)
        for row in cards.values():
            if not row_is_df(row):
                continue
            if not scope_matches_row(scope, row):
                continue
            _merge_card(all_cards, row)
        novos_globais = len(all_cards) - before_global
        count = len(cards)
        status = 'completo' if total is not None and total <= count else 'parcial'
        if total == 0:
            status = 'vazio'
        report.append({'oferta': oferta, 'escopo': scope.label, 'segmento': segment.label, 'total_informado': '' if total is None else str(total), 'cards_unicos': str(count), 'status': status, 'novos_globais': str(novos_globais), 'total_global_acumulado': str(len(all_cards)), 'urls_consultadas': ' | '.join(urls)})
        return (total, count)
    stop = False
    for oferta in ofertas:
        if stop:
            break
        for scope in build_scopes(config):
            if stop:
                break
            print(f'\n=== SEGMENTANDO {oferta}/{scope.label} ===', file=sys.stderr)
            harvest(oferta, scope, ListingSegment('imoveis'))
            for categoria in ['apartamentos', 'casas', 'comerciais', 'terrenos']:
                if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                    stop = True
                    break
                total_base, count_base = harvest(oferta, scope, ListingSegment(categoria))
                if total_base is not None and total_base <= 30:
                    continue
                if categoria in {'apartamentos', 'casas'}:
                    room_stats: list[tuple[str, int | None, int]] = []
                    for room_filter in ROOM_FILTERS:
                        if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                            stop = True
                            break
                        t, c = harvest(oferta, scope, ListingSegment(categoria, filtro=room_filter))
                        room_stats.append((room_filter, t, c))
                    if categoria == 'apartamentos' and (not stop):
                        for subtype in APARTMENT_SUBTYPES:
                            if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                                stop = True
                                break
                            harvest(oferta, scope, ListingSegment('apartamentos', subtipo=subtype))
                        for room_filter, room_total, room_count in room_stats:
                            if stop:
                                break
                            if room_total is None or room_total <= max(90, room_count):
                                continue
                            for subtype in APARTMENT_SUBTYPES:
                                if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                                    stop = True
                                    break
                                harvest(oferta, scope, ListingSegment('apartamentos', subtipo=subtype, filtro=room_filter))
                        if not stop:
                            for discovery_filter in APARTMENT_DISCOVERY_FILTERS:
                                if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                                    stop = True
                                    break
                                harvest(oferta, scope, ListingSegment('apartamentos', filtro=discovery_filter))
                    elif categoria == 'casas' and (not stop):
                        for discovery_filter in HOUSE_DISCOVERY_FILTERS:
                            if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                                stop = True
                                break
                            harvest(oferta, scope, ListingSegment('casas', filtro=discovery_filter))
                elif categoria == 'comerciais':
                    for subtype in COMMERCIAL_SUBTYPES:
                        if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                            stop = True
                            break
                        harvest(oferta, scope, ListingSegment('comerciais', subtipo=subtype))
                    if not stop:
                        for keyword in COMMERCIAL_KEYWORDS:
                            if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                                stop = True
                                break
                            harvest(oferta, scope, ListingSegment('comerciais', filtro=keyword))
                    if not stop:
                        for discovery_filter in COMMERCIAL_DISCOVERY_FILTERS:
                            if config.max_segmentos is not None and segment_counter[0] >= config.max_segmentos:
                                stop = True
                                break
                            harvest(oferta, scope, ListingSegment('comerciais', filtro=discovery_filter))
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(all_cards.values(), start=1):
        if config.limite and len(rows) >= config.limite:
            break
        if config.detalhes:
            try:
                print(f"Detalhe {idx}/{len(all_cards)}: {row['link']}", file=sys.stderr)
                detail_raw = fetch(session, row['link'], config.timeout, config.retries, config.backoff)
                row.update(parse_detail(detail_raw, row['link']))
                if row.pop('_area_range', ''):
                    row['area_util'] = ''
                if row.pop('_quartos_range', ''):
                    row['quartos'] = ''
                if row.pop('_price_consult', ''):
                    row['preco'] = ''
                    row['valor_m2'] = ''
                time.sleep(config.delay)
            except requests.RequestException as exc:
                print(f"  falha no detalhe {row['link']}: {exc}", file=sys.stderr)
        if not row_is_df(row):
            print(f"  ignorado fora do DF: {row['link']}", file=sys.stderr)
            continue
        matched = any((scope_matches_row(scope, row) for scope in build_scopes(config)))
        if not matched:
            print(f"  ignorado fora dos escopos selecionados: {row['link']} ({row.get('bairro')}/{row.get('cidade')})", file=sys.stderr)
            continue
        normalize_output_row(row)
        calculate_valor_m2(row)
        rows.append({field: row.get(field, '') for field in CSV_FIELDS})
        if checkpoint_path and len(rows) % 25 == 0:
            write_csv(rows, checkpoint_path)
    return (rows, report)

def scrape_normal(config: ScrapeConfig, arquivo_listagem: Path | None=None, checkpoint_path: Path | None=None) -> list[dict[str, str]]:
    session = make_session()
    ofertas = ['venda', 'aluguel'] if config.oferta == 'ambos' else [config.oferta]
    rows: list[dict[str, str]] = []
    seen_links: set[str] = set()
    if arquivo_listagem:
        raw = arquivo_listagem.read_text(encoding='utf-8', errors='replace')
        cards, _ = parse_listing(raw, ofertas[0], None, None, 1)
        for row in cards:
            normalize_output_row(row)
            calculate_valor_m2(row)
            rows.append({field: row.get(field, '') for field in CSV_FIELDS})
        return rows[:config.limite] if config.limite else rows
    for oferta in ofertas:
        for scope in build_scopes(config):
            url = build_listing_url(oferta, scope)
            print(f'Baixando listagem: {url}', file=sys.stderr)
            try:
                raw = fetch(session, url, config.timeout, config.retries, config.backoff)
            except requests.RequestException as exc:
                print(f'FALHA {oferta}/{scope.label}: {exc}', file=sys.stderr)
                continue
            if config.debug_html:
                save_debug_html(raw, f'{oferta}-{scope.label}', 1)
            cards, _ = parse_listing(raw, oferta, scope, url, 1)
            print(f'{oferta}/{scope.label}: {len(cards)} anúncios individuais encontrados', file=sys.stderr)
            for row in cards:
                if row['link'] in seen_links:
                    continue
                if config.detalhes:
                    try:
                        print(f"  detalhe: {row['link']}", file=sys.stderr)
                        detail_raw = fetch(session, row['link'], config.timeout, config.retries, config.backoff)
                        row.update(parse_detail(detail_raw, row['link']))
                        if row.pop('_area_range', ''):
                            row['area_util'] = ''
                        if row.pop('_quartos_range', ''):
                            row['quartos'] = ''
                        if row.pop('_price_consult', ''):
                            row['preco'] = ''
                            row['valor_m2'] = ''
                        time.sleep(config.delay)
                    except requests.RequestException as exc:
                        print(f"  falha no detalhe {row['link']}: {exc}", file=sys.stderr)
                if not row_is_df(row) or not scope_matches_row(scope, row):
                    continue
                normalize_output_row(row)
                calculate_valor_m2(row)
                seen_links.add(row['link'])
                rows.append({field: row.get(field, '') for field in CSV_FIELDS})
                if checkpoint_path:
                    write_csv(rows, checkpoint_path)
                if config.limite and len(rows) >= config.limite:
                    return rows
    return rows

def scrape_from_csv(config: ScrapeConfig, input_paths: list[Path], checkpoint_path: Path | None=None) -> list[dict[str, str]]:
    session = make_session()
    discovered: dict[str, dict[str, str]] = {}
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f'CSV de entrada não encontrado: {path}')
        with path.open('r', newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                link = (raw.get('link') or '').strip()
                if not link:
                    continue
                row = {field: raw.get(field) or '' for field in CSV_FIELDS}
                _merge_card(discovered, row)
    print(f'Entradas CSV: {len(input_paths)} arquivo(s) | {len(discovered)} anúncios únicos antes dos detalhes', file=sys.stderr)
    allowed_scopes = build_scopes(config)
    rows: list[dict[str, str]] = []
    total = len(discovered)
    for idx, row in enumerate(discovered.values(), start=1):
        if config.limite and len(rows) >= config.limite:
            break
        if config.detalhes:
            try:
                print(f"Detalhe {idx}/{total}: {row['link']}", file=sys.stderr)
                detail_raw = fetch(session, row['link'], config.timeout, config.retries, config.backoff)
                row.update(parse_detail(detail_raw, row['link']))
                if row.pop('_area_range', ''):
                    row['area_util'] = ''
                if row.pop('_quartos_range', ''):
                    row['quartos'] = ''
                if row.pop('_price_consult', ''):
                    row['preco'] = ''
                    row['valor_m2'] = ''
                time.sleep(config.delay)
            except requests.RequestException as exc:
                print(f"  falha no detalhe {row['link']}: {exc}", file=sys.stderr)
        if not row_is_df(row):
            continue
        if allowed_scopes and (not any((scope_matches_row(scope, row) for scope in allowed_scopes))):
            continue
        normalize_output_row(row)
        calculate_valor_m2(row)
        rows.append({field: row.get(field, '') for field in CSV_FIELDS})
        if checkpoint_path:
            write_csv(rows, checkpoint_path)
    return rows

def scrape(config: ScrapeConfig, arquivo_listagem: Path | None=None, checkpoint_path: Path | None=None, entrada_csv: list[Path] | None=None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if entrada_csv:
        if config.modo_segmentado:
            raise SystemExit('--entrada-csv não pode ser combinado com --modo-segmentado')
        if arquivo_listagem:
            raise SystemExit('--entrada-csv não pode ser combinado com --arquivo-listagem')
        return (scrape_from_csv(config, entrada_csv, checkpoint_path), [])
    if config.modo_segmentado and arquivo_listagem:
        raise SystemExit('--arquivo-listagem não pode ser combinado com --modo-segmentado')
    if config.modo_segmentado:
        return scrape_segmented(config, checkpoint_path)
    return (scrape_normal(config, arquivo_listagem, checkpoint_path), [])

def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Scraper de imóveis do WImoveis - somente DF.')
    parser.add_argument('--oferta', choices=['venda', 'aluguel', 'ambos'], default='ambos')
    parser.add_argument('--inicio', type=int, default=1, help='Mantido por compatibilidade; no modo atual apenas a página pública inicial é usada.')
    parser.add_argument('--fim', type=int, help='Mantido por compatibilidade; paginação protegida não é forçada.')
    parser.add_argument('--delay', type=float, default=1.0, help='Pausa entre requisições, em segundos.')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout HTTP, em segundos.')
    parser.add_argument('--retries', type=int, default=3, help='Tentativas extras em 403/429/5xx/timeouts.')
    parser.add_argument('--backoff', type=float, default=2.0, help='Base da espera exponencial entre tentativas.')
    parser.add_argument('--saida', help='CSV de saída. Padrão: AAAA-MM-DD.csv')
    parser.add_argument('--limite', type=int, help='Máximo de anúncios no total.')
    parser.add_argument('--sem-detalhes', action='store_true', help='Não abre a página individual.')
    parser.add_argument('--arquivo-listagem', type=Path, help='HTML local de listagem para teste.')
    parser.add_argument('--entrada-csv', action='append', type=Path, help='CSV de descoberta já existente. Pode repetir a opção para unir várias execuções e abrir os detalhes sem refazer a segmentação.')
    parser.add_argument('--escopo', choices=['bairros', 'cidades'], default='bairros', help='bairros percorre os bairros mapeados; cidades usa a cidade inteira.')
    parser.add_argument('--cidades', help='Lista separada por vírgula. Ex: brasilia,aguas claras')
    parser.add_argument('--bairros', help='Lista separada por vírgula nas cidades selecionadas.')
    parser.add_argument('--debug-html', action='store_true', help='Salva HTMLs em ./debug_html.')
    parser.add_argument('--modo-segmentado', action='store_true', help='Coleta várias sub-buscas públicas (tipo/quartos/subtipo/ordenação), deduplica por ID e evita a paginação que retorna HTTP 403.')
    parser.add_argument('--max-segmentos', type=int, help='Limita quantas URLs segmentadas serão consultadas; útil para teste.')
    parser.add_argument('--relatorio-segmentos', help='CSV do relatório de cobertura. Padrão: <saida>_segmentos.csv.')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.inicio < 1 or (args.fim is not None and args.fim < args.inicio):
        raise SystemExit('--inicio deve ser >= 1 e --fim deve ser >= --inicio')
    if args.max_segmentos is not None and args.max_segmentos < 1:
        raise SystemExit('--max-segmentos deve ser >= 1')
    config = ScrapeConfig(oferta=args.oferta, inicio=args.inicio, fim=args.fim, delay=args.delay, detalhes=not args.sem_detalhes, timeout=args.timeout, limite=args.limite, escopo=args.escopo, cidades=parse_csv_arg(args.cidades), bairros=parse_csv_arg(args.bairros), retries=args.retries, backoff=args.backoff, debug_html=args.debug_html, modo_segmentado=args.modo_segmentado, max_segmentos=args.max_segmentos)
    output_path = Path(args.saida) if args.saida else Path(f'{date.today().isoformat()}.csv')
    rows, report = scrape(config, args.arquivo_listagem, output_path, entrada_csv=args.entrada_csv)
    write_csv(rows, output_path)
    if args.modo_segmentado:
        if args.relatorio_segmentos:
            report_path = Path(args.relatorio_segmentos)
        else:
            report_path = output_path.with_name(f'{output_path.stem}_segmentos.csv')
        write_segment_report(report, report_path)
        completos = sum((1 for item in report if item.get('status') == 'completo'))
        parciais = sum((1 for item in report if item.get('status') == 'parcial'))
        print(f'Relatório de segmentos: {report_path} ({completos} completos, {parciais} parciais)')
    print(f'CSV gerado: {output_path} ({len(rows)} linhas)')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
