import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

def get_budgets_from_wikidata(imdb_ids):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    results = []

    for imdb_id in imdb_ids:
        query = f"""
        SELECT ?budget WHERE {{
          ?film wdt:P345 "{imdb_id}".
          ?film wdt:P2130 ?budget.
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            data = sparql.query().convert()
            bindings = data["results"]["bindings"]
            if bindings:
                budget_value = bindings[0]["budget"]["value"]
                results.append((imdb_id, float(budget_value)))
            else:
                results.append((imdb_id, None))
        except:
            results.append((imdb_id, None))
    
    return pd.DataFrame(results, columns=['tconst', 'budget'])
