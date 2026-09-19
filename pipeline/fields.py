"""Canonical field vocabulary for the SI-vs-BL comparison.

The SI and BL render the same field under different labels (e.g. "Port of
Loading" vs "Load Port"). These patterns match a raw label string (already
lowercased) to the canonical field it refers to, regardless of which synonym
was used.
"""
import re

COMPARE_FIELDS = [
    "shipper", "consignee", "notify_party",
    "port_of_loading", "port_of_discharge",
    "container_count", "gross_weight_kg",
]

# Extra fields we extract but don't score on -- useful for wrong-doc-type
# detection (a real SI/BL mentions vessel/commodity; an invoice/COO doesn't).
AUX_FIELDS = ["vessel", "commodity"]

ALL_FIELDS = COMPARE_FIELDS + AUX_FIELDS

FIELD_PATTERNS = {
    "shipper": re.compile(r"\bshipper\b"),
    "consignee": re.compile(r"\bconsignee\b|\bto the order of\b"),
    "notify_party": re.compile(r"\bnotify\b"),
    "port_of_loading": re.compile(r"\bport of loading\b|\bload port\b|\bpol\b"),
    "port_of_discharge": re.compile(r"\bport of discharge\b|\bdischarge port\b|\bpod\b"),
    "container_count": re.compile(r"\bcontainer"),
    "gross_weight_kg": re.compile(r"\bgross w"),
    "vessel": re.compile(r"\bvessel\b"),
    "commodity": re.compile(r"\bcommodity\b|\bdescription\b"),
}

# Checked in this order so a more specific pattern wins before a looser one
# can shadow it. Notify-party labels ("Notify Party/Intermediate Consignee")
# can contain the substring "consignee", so notify_party must be checked
# first -- no consignee label ever contains "notify".
FIELD_ORDER = [
    "notify_party", "port_of_discharge", "port_of_loading", "consignee",
    "shipper", "container_count", "gross_weight_kg", "vessel", "commodity",
]

BLANK_TOKENS = {"???", "_______", "TBA", "TBC", "N/A", "____MT", "", "_", "__"}

# A matched label is only accepted if its value looks plausible for that
# field -- guards against a mis-split table header (e.g. "CONTAINER NO.
# DESCRIPTION" / "GROSS WEIGHT (KG)") shadowing the real value that appears
# later in the document.
VALUE_HINTS = {
    "container_count": re.compile(r"\d+\s*[xX]\s*\S"),
    "gross_weight_kg": re.compile(r"\d"),
}


def value_plausible(field, value):
    hint = VALUE_HINTS.get(field)
    if hint is None:
        return True
    return bool(hint.search(str(value or "")))


def match_field(raw_label):
    """Return the canonical field name a raw label string refers to, or None."""
    low = raw_label.strip().lower()
    for field in FIELD_ORDER:
        if FIELD_PATTERNS[field].search(low):
            return field
    return None


def is_blank_value(value):
    if value is None:
        return True
    if not isinstance(value, str):
        return False  # a numeric cell (e.g. xlsx gross weight) is never blank
    v = value.strip().strip(".").strip()
    if v in BLANK_TOKENS:
        return True
    stripped = v.strip("_?").strip()
    return stripped == "" and v != ""
