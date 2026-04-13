import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

TRAIN_DATA = [
    ("This agreement is between LexiCorp and Smith & Co on 2026-04-13.", 
     {"entities": [(26, 34, "PARTY"), (39, 49, "PARTY"), (53, 63, "DATE")]}),
    ("The total termination fee is $50,000.", 
     {"entities": [(30, 37, "MONEY")]})
]

# Load a blank English model or a pre-trained legal model
nlp = spacy.blank("en") 
db = DocBin()

for text, annot in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("./train.spacy")
# Run in terminal: python -m spacy train config.cfg --output ./output
# Example of annotated training data