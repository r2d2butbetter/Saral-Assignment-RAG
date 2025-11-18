import re


def extract_cited_sources(text):
    """Extract cited source numbers from text."""
    return set(re.findall(r'\[Source (\d+)\]', text))
