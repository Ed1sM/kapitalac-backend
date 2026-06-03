import re
from pathlib import Path
from typing import Optional

import pdfplumber


FINANCIAL_FIELDS = {
    # Bilans stanja
    "fixed_assets": "002",
    "current_assets": "025",
    "inventory": "026",
    "short_term_receivables": "031",
    "cash": "043",
    "total_assets": "046",
    "equity": "101",
    "retained_earnings": "111",
    "retained_profit_previous_years": "112",
    "retained_profit_current_year": "113",
    "loss_previous_years": "114",
    "loss_current_year": "115",
    "long_term_liabilities": "122",
    "short_term_liabilities": "129",
    "total_liabilities_and_equity": "144",

    # Bilans uspjeha
    "sales_revenue": "201",
    "operating_profit": "221",
    "financial_result": "241",
    "profit_before_tax": "244",
    "net_profit": "248",

    # Tokovi gotovine
    "operating_cash_flow": "311",
    "cash_end_period": "337",

    # Statistički aneks
    "average_employees": "001",
    "sales_revenue_goods": "002",
    "sales_revenue_products_services": "003",
}


def extract_text_from_pdf(pdf_path: str) -> str:
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1, y_tolerance=3)
            if text:
                all_text.append(text)

    return "\n".join(all_text)


def parse_number(value: Optional[str]):
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    value = value.replace(".", "")
    value = value.replace(",", ".")

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except ValueError:
        return None


def clean_company_name(name: str) -> str:
    name = name.replace("\n", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def find_company_name(text: str) -> Optional[str]:
    lines = text.splitlines()

    for index, line in enumerate(lines):
        if "Naziv:" not in line:
            continue

        company_parts = [line.split("Naziv:", 1)[1].strip()]

        for next_line in lines[index + 1:index + 5]:
            clean_line = next_line.strip()

            stop_words = [
                "Popunjava",
                "Sjedište:",
                "Matični broj:",
                "Šifra djelatnosti:",
                "Grupa",
                "Pozicija",
                "AKTIVA",
                "PASIVA",
            ]

            if any(clean_line.startswith(word) for word in stop_words):
                break

            if clean_line:
                company_parts.append(clean_line)

        return clean_company_name(" ".join(company_parts))

    return None


def find_meta_data(text: str) -> dict:
    meta = {
        "company_name": None,
        "registration_number": None,
        "activity_code": None,
        "report_year": None,
    }

    registration_match = re.search(r"Matični broj:\s*(\d+)", text)
    if registration_match:
        meta["registration_number"] = registration_match.group(1)

    activity_match = re.search(r"Šifra djelatnosti:\s*(\d+)", text)
    if activity_match:
        meta["activity_code"] = activity_match.group(1)

    year_match = re.search(r"31\.12\.(\d{4})", text)
    if year_match:
        meta["report_year"] = int(year_match.group(1))

    meta["company_name"] = find_company_name(text)

    return meta


def extract_section(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    start_index = text.find(start_marker)

    if start_index == -1:
        return ""

    if end_marker is None:
        return text[start_index:]

    end_index = text.find(end_marker, start_index + len(start_marker))

    if end_index == -1:
        return text[start_index:]

    return text[start_index:end_index]


def find_values_by_row_code(section_text: str, row_code: str) -> dict:
    number_pattern = r"-?\d{1,3}(?:\.\d{3})*(?:,\d+)?|-?\d+"

    valid_row_pattern = re.compile(
        rf"\b{row_code}\b\s+({number_pattern})(?:\s+({number_pattern}))?\s*$"
    )

    for line in section_text.splitlines():
        line = line.strip()

        if not line:
            continue

        match = valid_row_pattern.search(line)

        if not match:
            continue

        current_year = parse_number(match.group(1))
        previous_year = parse_number(match.group(2)) if match.group(2) else None

        return {
            "current_year": current_year,
            "previous_year": previous_year,
            "source_line": line,
        }

    return {
        "current_year": None,
        "previous_year": None,
        "source_line": None,
    }


def zero_if_none(value):
    return 0 if value is None else value


def set_fallback_detail(result: dict, field: str, row_code: str, value, source_line: str):
    result[field] = value
    result[f"{field}_previous"] = None

    result["extraction_details"][field] = {
        "row_code": row_code,
        "current_year": value,
        "previous_year": None,
        "source_line": source_line,
    }


def apply_fallbacks(result: dict) -> dict:
    """
    Fallback pravila za crnogorske finansijske izvještaje.

    U zvaničnim PDF izvještajima prazna polja često znače 0.
    Zato neke nedostajuće vrijednosti postavljamo na 0 ili ih računamo
    iz drugih pozicija bilansa.
    """

    # 1. Ako ukupna aktiva nije pronađena, ali ukupna pasiva jeste,
    # koristimo ukupnu pasivu kao ukupnu aktivu jer bilans mora biti izjednačen.
    if result.get("total_assets") is None and result.get("total_liabilities_and_equity") is not None:
        set_fallback_detail(
            result=result,
            field="total_assets",
            row_code="046_fallback",
            value=result.get("total_liabilities_and_equity"),
            source_line="Fallback: total_assets = total_liabilities_and_equity",
        )

    # 2. Ako nema dugoročnih obaveza, tretiramo ih kao 0.
    if result.get("long_term_liabilities") is None:
        set_fallback_detail(
            result=result,
            field="long_term_liabilities",
            row_code="122_fallback",
            value=0,
            source_line="Fallback: prazna pozicija dugoročnih obaveza tretirana kao 0",
        )

    # 3. Ako nema obrtnih sredstava, računamo ih kao:
    # obrtna sredstva = ukupna aktiva - stalna imovina.
    if (
        result.get("current_assets") is None
        and result.get("total_assets") is not None
        and result.get("fixed_assets") is not None
    ):
        current_assets = result.get("total_assets") - result.get("fixed_assets")

        set_fallback_detail(
            result=result,
            field="current_assets",
            row_code="025_fallback",
            value=current_assets,
            source_line="Fallback: current_assets = total_assets - fixed_assets",
        )

    # 4. Ako kratkoročne obaveze nijesu pronađene, pokušavamo ih izračunati:
    # ukupne obaveze = ukupna aktiva - kapital
    # kratkoročne obaveze = ukupne obaveze - dugoročne obaveze
    if (
        result.get("short_term_liabilities") is None
        and result.get("total_assets") is not None
        and result.get("equity") is not None
    ):
        total_liabilities = result.get("total_assets") - result.get("equity")
        short_term_liabilities = total_liabilities - zero_if_none(result.get("long_term_liabilities"))

        set_fallback_detail(
            result=result,
            field="short_term_liabilities",
            row_code="129_fallback",
            value=short_term_liabilities,
            source_line=(
                "Fallback: short_term_liabilities = "
                "(total_assets - equity) - long_term_liabilities"
            ),
        )

    # 5. Fallback za retained_earnings:
    # 112 + 113 - 114 - 115
    if result.get("retained_earnings") is None:
        retained_parts = [
            result.get("retained_profit_previous_years"),
            result.get("retained_profit_current_year"),
            result.get("loss_previous_years"),
            result.get("loss_current_year"),
        ]

        if any(value is not None for value in retained_parts):
            retained_earnings = (
                zero_if_none(result.get("retained_profit_previous_years"))
                + zero_if_none(result.get("retained_profit_current_year"))
                - zero_if_none(result.get("loss_previous_years"))
                - zero_if_none(result.get("loss_current_year"))
            )

            set_fallback_detail(
                result=result,
                field="retained_earnings",
                row_code="111_fallback",
                value=retained_earnings,
                source_line="Fallback: retained_earnings = 112 + 113 - 114 - 115",
            )

    # 6. Fallback za sales_revenue:
    # prvo pokušavamo iz Statističkog aneksa 002 + 003.
    if result.get("sales_revenue") is None:
        goods = result.get("sales_revenue_goods")
        services = result.get("sales_revenue_products_services")

        if goods is not None or services is not None:
            sales_revenue = zero_if_none(goods) + zero_if_none(services)

            set_fallback_detail(
                result=result,
                field="sales_revenue",
                row_code="201_fallback",
                value=sales_revenue,
                source_line="Fallback: sales_revenue = statistički aneks red 002 + red 003",
            )

    # 7. Ako sales_revenue i dalje nije pronađen, tretiramo ga kao 0.
    # Kod dosta malih/rizičnih firmi red prihoda je prazan jer firma nije imala promet.
    if result.get("sales_revenue") is None:
        set_fallback_detail(
            result=result,
            field="sales_revenue",
            row_code="201_zero_fallback",
            value=0,
            source_line="Fallback: prazna pozicija prihoda od prodaje tretirana kao 0",
        )

    # 8. Ako nema zaliha, gotovine ili potraživanja, tretiramo prazno kao 0.
    zero_fallback_fields = {
        "inventory": "026",
        "short_term_receivables": "031",
        "cash": "043",
    }

    for field, row_code in zero_fallback_fields.items():
        if result.get(field) is None:
            set_fallback_detail(
                result=result,
                field=field,
                row_code=f"{row_code}_zero_fallback",
                value=0,
                source_line=f"Fallback: prazna pozicija {field} tretirana kao 0",
            )

    return result


def extract_financials_from_pdf(pdf_path: str) -> dict:
    text = extract_text_from_pdf(pdf_path)
    meta = find_meta_data(text)

    balance_sheet = extract_section(
        text,
        "ISKAZ O FINANSIJSKOJ POZICIJI /BILANS STANJA/",
        "ISKAZ O UKUPNOM REZULTATU /BILANS USPJEHA/",
    )

    income_statement = extract_section(
        text,
        "ISKAZ O UKUPNOM REZULTATU /BILANS USPJEHA/",
        "ISKAZ O TOKOVIMA GOTOVINE",
    )

    cash_flow = extract_section(
        text,
        "ISKAZ O TOKOVIMA GOTOVINE",
        "ISKAZ O PROMJENAMA NA KAPITALU",
    )

    statistical_annex = extract_section(
        text,
        "STATISTIČKI ANEKS",
        "OBRAČUN AMORTIZACIJE",
    )

    sections = {
        # Bilans stanja
        "fixed_assets": balance_sheet,
        "current_assets": balance_sheet,
        "inventory": balance_sheet,
        "short_term_receivables": balance_sheet,
        "cash": balance_sheet,
        "total_assets": balance_sheet,
        "equity": balance_sheet,
        "retained_earnings": balance_sheet,
        "retained_profit_previous_years": balance_sheet,
        "retained_profit_current_year": balance_sheet,
        "loss_previous_years": balance_sheet,
        "loss_current_year": balance_sheet,
        "long_term_liabilities": balance_sheet,
        "short_term_liabilities": balance_sheet,
        "total_liabilities_and_equity": balance_sheet,

        # Bilans uspjeha
        "sales_revenue": income_statement,
        "operating_profit": income_statement,
        "financial_result": income_statement,
        "profit_before_tax": income_statement,
        "net_profit": income_statement,

        # Tokovi gotovine
        "operating_cash_flow": cash_flow,
        "cash_end_period": cash_flow,

        # Statistički aneks
        "average_employees": statistical_annex,
        "sales_revenue_goods": statistical_annex,
        "sales_revenue_products_services": statistical_annex,
    }

    result = {
        "file_name": Path(pdf_path).name,
        **meta,
    }

    extraction_details = {}

    for field_name, row_code in FINANCIAL_FIELDS.items():
        values = find_values_by_row_code(sections[field_name], row_code)

        result[field_name] = values["current_year"]
        result[f"{field_name}_previous"] = values["previous_year"]

        extraction_details[field_name] = {
            "row_code": row_code,
            "current_year": values["current_year"],
            "previous_year": values["previous_year"],
            "source_line": values["source_line"],
        }

    result["extraction_details"] = extraction_details

    result = apply_fallbacks(result)

    return result