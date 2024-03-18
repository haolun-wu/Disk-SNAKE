# %%
import tqdm
import requests
import os
import pandas as pd
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.sparql import (
    # return_sparql_query_results,
    get_subclasses_of_item,
)


human_fields = {
    "name": "itemLabel",
    "dob": "dob",
    "pob": "pobLabel",
    "dod": "dod",
    "pod": "podLabel",
    "educated_at": "educatedAtLabel",
    "occupation": "occupationLabel",
    "gender": "genderLabel",
    "country": "countryLabel",
    "ethnic_group": "ethnicGroupLabel",
    "religion": "religionLabel",
    "known_for": "knownForLabel",
    "cause_of_death": "causeOfDeathLabel",
    "place_of_burial": "placeOfBurialLabel",
}


human_optionals = """
        OPTIONAL {{ ?item wdt:P569 ?dob. }}
        OPTIONAL {{ ?item wdt:P19 ?pob. }}
        OPTIONAL {{ ?item wdt:P570 ?dod. }}
        OPTIONAL {{ ?item wdt:P20 ?pod. }}
        OPTIONAL {{ ?item wdt:P69 ?educatedAt. }}
        OPTIONAL {{ ?item wdt:P106 ?occupation. }}
        OPTIONAL {{ ?item wdt:P21 ?gender. }}
        OPTIONAL {{ ?item wdt:P27 ?country. }}
        OPTIONAL {{ ?item wdt:P172 ?ethnicGroup. }}
        OPTIONAL {{ ?item wdt:P140 ?religion. }}
        OPTIONAL {{ ?item wdt:P800 ?knownFor. }}
        OPTIONAL {{ ?item wdt:P509 ?causeOfDeath. }}
        OPTIONAL {{ ?item wdt:P119 ?placeOfBurial. }}
        """


def id_to_label(id):
    return (
        get_entity_dict_from_api(id)["labels"]
        .get("en", {})
        .get("value", "<na>")
    )


def construct_query(item_class, optionals, fields):
    query = f"SELECT ?item ?{' ?'.join(fields.values())}\n"
    query += "WHERE {{" + f"?item wdt:P31 wd:{item_class} ."
    query += optionals
    query += """
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10000
        """
    return query


def get_wikidata(item_class, fields, download=False):
    class_label = id_to_label(item_class).replace(" ", "_")
    fpath = f"data/wikidata-{class_label}({item_class}).csv"
    if download or not os.path.exists(fpath):
        # Define the SPARQL query to retrieve the data
        query = construct_query(item_class, human_optionals, fields)
        data = fetch_paginated_results(query)
        rows = [
            {
                key: item.get(wikidata_key, {}).get("value", "<na>")
                for key, wikidata_key in fields.items()
            }
            for item in data
        ]
        # Convert the results into a new pandas dataframe
        df = pd.DataFrame(rows, columns=fields.keys())
        # Save the dataframe as a CSV file
        os.makedirs("data", exist_ok=True)
        df.to_csv(fpath, index=False)
    df = pd.read_csv(fpath)
    return df


def send_query(query):
    endpoint_url = "https://query.wikidata.org/sparql"
    # headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(
        endpoint_url,
        params={"format": "json", "query": query, "timeout": "600000"},
        headers={"User-Agent": "MyApp"},
    )
    return response.json()


def fetch_paginated_results(base_query, limit=100, max_pages=float("inf")):
    offset = 0
    has_more_results = True
    all_results = []
    while has_more_results:
        query = base_query.format(limit=limit, offset=offset)
        data = send_query(query)

        results = data["results"]["bindings"]
        all_results.extend(results)

        has_more_results = len(results) == limit and offset < limit * max_pages
        offset += limit

    return all_results


if __name__ == "__main__":
    dfs = []
    subclasses = tqdm.tqdm(get_subclasses_of_item("Q901")[1:])
    for subclass in subclasses:
        subclasses.set_description(
            f"Processing {subclass} ({id_to_label(subclass)})"
        )
        df = get_wikidata(item_class=subclass, fields=human_fields)
        if not df.empty:
            df_small = (
                df.groupby(["name", "dob", "dod", "pob", "pod"])
                .agg(lambda x: list(set(x)))
                .reset_index()
            )
            dfs.append(df_small)
    df = pd.concat(dfs)
    save_path = f"data/wikidata-Q901({id_to_label('Q901')}).csv"
