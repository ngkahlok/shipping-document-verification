"""Value normalization so the same field compares equal across formats.

Different renderers format the same field differently even with no defect:
  - ports carry a trailing "(LOCODE)" in .txt but not always in .pdf
  - entity fields carry "Name | address..." in .xlsx, "Name\\naddress" in
    .docx, and a bare name in .txt
  - weights are a formatted "21,577 KG" string in .txt but a raw int cell
    in .xlsx
These normalizers strip that formatting noise so only the real value remains.
"""
import re


def _entity_name(value):
    """First line, before any pipe-joined address, of an entity field."""
    v = (value or "").strip()
    v = v.split("|")[0]
    v = v.splitlines()[0]
    return re.sub(r"\s+", " ", v).strip().upper()


def _place(value):
    """Strip parenthetical codes/sub-names and punctuation from a port name."""
    v = str(value or "")
    v = re.sub(r"\([^)]*\)", " ", v)  # drop any "(...)" content
    v = re.sub(r"[^A-Za-z0-9]+", " ", v)
    return re.sub(r"\s+", " ", v).strip().upper()


def _digits(value):
    """Pull the numeric content out of a weight/count string."""
    s = re.sub(r"[^0-9]", "", str(value or ""))
    return s.lstrip("0") or ("0" if s else "")


def _container(value):
    """'1 x 40'HC' -> ('1', "40'HC") so count and size compare separately."""
    s = str(value or "")
    m = re.match(r"\s*(\d+)\s*[xX]\s*(.+)", s)
    if not m:
        return _digits(s), ""
    count, size = m.groups()
    size = re.sub(r"[^A-Za-z0-9]+", "", size).upper()
    return _digits(count), size


NORMALIZERS = {
    "shipper": _entity_name,
    "consignee": _entity_name,
    "notify_party": _entity_name,
    "port_of_loading": _place,
    "port_of_discharge": _place,
    "gross_weight_kg": _digits,
}


def normalized_value(field, raw_value):
    if field == "container_count":
        return _container(raw_value)
    fn = NORMALIZERS.get(field)
    return fn(raw_value) if fn else str(raw_value or "").strip().upper()


def values_match(field, raw_a, raw_b):
    return normalized_value(field, raw_a) == normalized_value(field, raw_b)
