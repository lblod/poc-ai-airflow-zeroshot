import uuid

import fire
import requests
from tqdm import tqdm

from gcs import read_json

# Still hardcoded, when moving to production this could be improved... by using more intuitive uris
bbc_uri_mapping = [{'bbc_lvl1': 'Algemene financiering',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/b595897a-3a3e-406e-840d-69fa7020fc85'},
                   {'bbc_lvl1': 'Algemeen bestuur',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/31ef2c84-8779-4d97-ab45-c460d8780034'},
                   {'bbc_lvl1': 'Zich verplaatsen en mobiliteit',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/87adadcd-b5ef-4c89-9ca8-b4d7c0bf4e13'},
                   {'bbc_lvl1': 'Natuur en milieubeheer',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/5c616d64-fe79-4eb3-a024-a6a31f72333e'},
                   {'bbc_lvl1': 'Veiligheidszorg',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/c7776fbb-a32a-419c-b996-4dc8a089e387'},
                   {'bbc_lvl1': 'Ondernemen en werken',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/e25aa915-c026-4529-9137-4f63ded4d168'},
                   {'bbc_lvl1': 'Wonen en ruimtelijke ordening',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/b06c09cc-71ad-41a5-a942-e0fa29427f13'},
                   {'bbc_lvl1': 'Cultuur en vrije tijd',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/4cec000d-2403-470e-965e-26cefba58cf3'},
                   {'bbc_lvl1': 'Leren en onderwijs',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/a4aed4e7-7a70-4481-a374-eec3f15bd09b'},
                   {'bbc_lvl1': 'Zorg en opvang',
                    'uri': 'http://data.lblod.info/ML2GrowClassification/f60a87e6-f262-470e-9c23-c6caf7e9848a'}]

new_mapping = {bbc["bbc_lvl1"]: bbc["uri"] for bbc in bbc_uri_mapping}

headers = {
    "Accept": "application/sparql-results+json,*/*;q=0.9"
}


def main(endpoint):
    """
    Function that creates and executes the DELETE WHERE and INSERT statements

    :param endpoint: url to the sparql endpoint
    :return:
    """

    # load the file containing the classified content
    records = read_json(file_name="classified.json")

    # If no records are available skip
    if not records:
        return None

    for record in tqdm(records):
        try:
            file_name = record["thing"]
            uris, query_extension = [], []
            n = {new_mapping[k]: v for k, v in record["bbc"].items()}
            for k, v in n.items():
                uri = f"http://data.lblod.info/ML2GrowClassification/score/{str(uuid.uuid4())}"
                uris.append(f"<{uri}>")

                query_extension.append(
                    f"""<{uri}> a <{k}> ;
                    ext:score {v} .
                    """
                )

            q = f"""
                PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            
                DELETE {{
                GRAPH <http://mu.semte.ch/application> {{
                 <{file_name}> ext:BBC_scoring ?bbc .
                 <{file_name}> ext:ingestedMl2GrowSmartRegulationsBBC ?sbr .
                 ?bbc ext:score ?bbc_score .
                    }}
                }}
                
                WHERE {{
                 <{file_name}> ext:BBC_scoring ?bbc .
                 <{file_name}> ext:ingestedMl2GrowSmartRegulationsBBC ?sbr .
                 ?bbc ext:score ?bbc_score .
            }}
            """

            # DELETE {{
            #     GRAPH <http://mu.semte.ch/application> {{
            #         <{file_name}> ext:BBC_scoring ?oldscore .
            #     }}
            # }}
            # WHERE {{
            #     GRAPH <http://mu.semte.ch/application> {{
            #         OPTIONAL {{ <{file_name}> ext:BBC_scoring ?oldscore . }}
            #     }}
            # }}

            # request for the sparql DELETE
            r = requests.post(endpoint, data={"query": q}, headers=headers)

            if r.status_code != 200:
                print(f"[FAILURE] {50 * '-'} /n {q} /n {50 * '-'}")

            q = f"""
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
    
            INSERT {{
            GRAPH <http://mu.semte.ch/application> {{
    
                <{file_name}> ext:BBC_scoring {' , '.join(uris)} .
                <{file_name}> ext:ingestedMl2GrowSmartRegulationsBBC "1".
                {''.join(query_extension)}
    
                }}
            }}
    
            """

            # request for the sparql INSERT
            r = requests.post(endpoint, data={"query": q}, headers=headers)

            if r.status_code != 200:
                print(f"[FAILURE] {50 * '-'} /n {q} /n {50 * '-'}")
        except Exception as ex:
            print(r.text)
            print(ex)


if __name__ == '__main__':
    fire.Fire(main)
