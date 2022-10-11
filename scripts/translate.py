import fire
import pandas as pd
import torch
from nltk import tokenize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline

from dataset import Dataset_from_DF
from gcs import read_json, write_json


class Translator:

    def __init__(self, translation_model: str, df: pd.DataFrame):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.__initialize(translation_model, df)

    def __initialize(self, model_name, df):
        # Initialize current model
        model_name = "/models/translate-nl-en"
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)

        # Create training dataset
        train_set = Dataset_from_DF(
            dataframe=df,
            tokenizer=self.tokenizer,
            max_length=512,
            device=self.device,
            text_column="text"  # hardcoded for now
        )

        self.data_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    def __call__(self):
        resulting_files = []
        with torch.no_grad():
            for data in tqdm(self.data_loader):
                output = self.model.generate(**{k: v.squeeze(0) for k, v in data.items()})
                v = self.tokenizer.batch_decode(
                    output,
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True
                )
                resulting_files.append(" ".join(v))

        return resulting_files


def new_translate(model):
    records = read_json(file_name="export.json")
    df = pd.DataFrame(records)
    translate = Translator(model, df)
    df["english"] = translate()
    write_json(file_name="translated.json", content=df.to_dict(orient="records"))


def handle(translator, corpus):
    try:
        translated = []
        for sentence in tokenize.sent_tokenize(corpus):
            translated.append(translator(sentence)[0]['translation_text'])
        return " ".join(translated)
    except:
        return None


def translate(model):
    records = read_json(file_name="export.json")
    nl_en_translator = pipeline("translation_nl_to_en", model="/models/translate-nl-en")
    updated = [
        {
            **entry,
            'english': handle(nl_en_translator, entry['text'][:10_000])
        }
        for entry in tqdm(records)
    ]
    write_json(file_name="translated.json", content=updated)


if __name__ == '__main__':
    fire.Fire(translate)
