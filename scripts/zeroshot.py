import fire
from tqdm import tqdm
from transformers import pipeline

from gcs import read_json, write_json


def classify():
    """
    This function does the zeroshot classification

    :return:
    """

    # Load bbc taxonomy (candidate labels)
    candid_json = read_json(file_name="bbc_taxo.json")
    candidate_labels = [candid["nl"] for candid in candid_json]

    # initialize the huggingface pipieline for zeroshot classifiction with the model out of the moder_store
    classifier = pipeline("zero-shot-classification", model="/models/zero-shot-model")

    # Load the text content to do inference on.
    records = read_json(file_name="export.json")

    # create empty list for the updated content
    updated = []

    # loop over the records containing the input text and execute the zeroshot learning approach.
    for entry in tqdm(records):
        try:
            # execute the zeroshot on the given text.
            result = classifier(entry['text'].lower(), candidate_labels)

            # zip found label and confidence
            labels = {x[0]: x[1] for x in zip(result["labels"], result["scores"])}  # if x[1] > 0.2

            # append results to the record that just got processed for next steps.
            updated.append({**entry, "bbc": labels})
        except Exception as ex:
            print(ex)

    # write the content back to a file for next steps
    write_json(file_name="classified.json", content=updated)


if __name__ == '__main__':
    fire.Fire(classify)
