import fire
from tqdm import tqdm
from transformers import pipeline

from gcs import read_json, write_json


def classify():
    candid_json = read_json(file_name="bbc_taxo.json")
    candidate_labels = [candid["nl"] for candid in candid_json]
    classifier = pipeline("zero-shot-classification", model="/models/zero-shot-model")
    records = read_json(file_name="export.json")
    updated = []
    for entry in tqdm(records):
        try:
            result = classifier(entry['text'].lower(), candidate_labels)
            labels = {x[0]:x[1] for x in zip(result["labels"], result["scores"])}  # if x[1] > 0.2
            updated.append({**entry, "bbc": labels})
        except:
            pass
    write_json(file_name="classified.json", content=updated)


if __name__ == '__main__':
    fire.Fire(classify)
