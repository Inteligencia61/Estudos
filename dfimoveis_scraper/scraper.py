from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


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


def fetch(session: requests.Session, url: str, timeout: int) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"
    return response.text


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
    arquivo_listagem: Path | None = None,
    checkpoint_path: Path | None = None,
) -> list[dict[str, str]]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    )

    ofertas = ["venda", "aluguel"] if config.oferta == "ambos" else [config.oferta]
    rows: list[dict[str, str]] = []
    seen_links: set[str] = set()

    def add_cards(cards: list[dict[str, str]]) -> bool:
        for row in cards:
            if row["link"] in seen_links:
                continue
            seen_links.add(row["link"])
            if config.detalhes and not arquivo_listagem:
                try:
                    print(f"Baixando detalhe: {row['link']}", file=sys.stderr)
                    detail = parse_detail(fetch(session, row["link"], config.timeout))
                    row.update(detail)
                    time.sleep(config.delay)
                except requests.RequestException as exc:
                    print(f"Falha no detalhe {row['link']}: {exc}", file=sys.stderr)
            rows.append({field: row.get(field, "") for field in CSV_FIELDS})
            if checkpoint_path:
                write_csv(rows, checkpoint_path)
            if config.limite and len(rows) >= config.limite:
                return True
        return False

    if arquivo_listagem:
        raw = arquivo_listagem.read_text(encoding="utf-8", errors="replace")
        cards = parse_listing(raw)
        print(f"{arquivo_listagem}: {len(cards)} imoveis encontrados", file=sys.stderr)
        add_cards(cards)
        return rows

    for oferta in ofertas:
        for scope in build_scopes(config):
            pagina = config.inicio
            while True:
                if config.fim is not None and pagina > config.fim:
                    break
                url = build_listing_url(oferta, pagina, scope)
                print(f"Baixando listagem: {url}", file=sys.stderr)
                try:
                    raw = fetch(session, url, config.timeout)
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        print(f"Parando {oferta}/{scope.label}: pagina {pagina} retornou 404.", file=sys.stderr)
                        break
                    raise
                cards = parse_listing(raw)
                print(f"{url}: {len(cards)} imoveis encontrados", file=sys.stderr)
                if not cards:
                    print(f"Parando {oferta}/{scope.label}: pagina {pagina} sem imoveis.", file=sys.stderr)
                    break
                if add_cards(cards):
                    return rows
                pagina += 1
                time.sleep(config.delay)

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
    parser.add_argument("--saida", help="Arquivo CSV de saida. Se omitido, usa a data da coleta.")
    parser.add_argument("--limite", type=int, help="Numero maximo de imoveis para capturar.")
    parser.add_argument("--sem-detalhes", action="store_true", help="Nao abrir paginas individuais dos imoveis.")
    parser.add_argument("--arquivo-listagem", type=Path, help="HTML local de uma listagem para teste.")
    parser.add_argument(
        "--escopo",
        choices=["bairros", "todos"],
        default="bairros",
        help="Modo bairros percorre cidade/bairro; todos usa a URL antiga /df/todos.",
    )
    parser.add_argument("--cidades", help="Lista separada por virgula. Ex: brasilia,aguas claras.")
    parser.add_argument("--bairros", help="Lista separada por virgula para usar nas cidades selecionadas.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inicio < 1 or (args.fim is not None and args.fim < args.inicio):
        raise SystemExit("--inicio deve ser >= 1 e --fim deve ser >= --inicio")

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
    )
    output_path = Path(args.saida) if args.saida else Path(f"{date.today().isoformat()}.csv")
    rows = scrape(config, args.arquivo_listagem, output_path)
    write_csv(rows, output_path)
    print(f"CSV gerado: {output_path} ({len(rows)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
