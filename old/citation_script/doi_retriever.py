import requests
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bparser import BibTexParser

def find_doi(title):
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 1}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        items = results.get("message", {}).get("items", [])
        if items:
            return items[0].get("DOI")
    return None

# Load your original .bib file
with open('yourfile.bib') as bibtex_file:
    parser = BibTexParser(common_strings=True)
    bib_database = bibtexparser.load(bibtex_file, parser=parser)

# Search for DOIs and update entries
for entry in bib_database.entries:
    if "doi" not in entry:  # Check if DOI is already present
        title = entry.get("title")
        if title:
            doi = find_doi(title)
            if doi:
                entry["doi"] = doi
                print(f'Added DOI for {title}: {doi}')

# Write the updated entries to a new .bib file
with open('updatedfile.bib', 'w') as bibtex_file:
    bibtexwriter = BibTexWriter()
    bibtex_file.write(bibtexwriter.write(bib_database))
