from collections import defaultdict
import pickle
import os
from rdflib import RDFS, Graph, Namespace
from rdflib.plugins.sparql import prepareQuery
from embedding_handler import EmbeddingHandler

class GraphProcessor:
    KG_GRAPH_PATH = './../Dataset/14_graph.nt'
    CACHE_GRAPH_PATH = './../Dataset/graph.pkl'
    
    # Namespaces
    WD = Namespace('http://www.wikidata.org/entity/')
    WDT = Namespace('http://www.wikidata.org/prop/direct/')
    SCHEMA = Namespace('http://schema.org/')
    DDIS = Namespace('http://ddis.ch/atai/')
    RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')

    def __init__(self, dataset_path= KG_GRAPH_PATH, cache_graph_path= CACHE_GRAPH_PATH):
        self.graph = Graph()
        
        self._get_graph_cache(dataset_path, cache_graph_path)
        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(RDFS.label)}
        self.embedding_handler = EmbeddingHandler(graph=self.graph, ent2lbl=self.ent2lbl)
    
    def get_answer_by_embedding(self, best_matched_movie, user_query):
        try:
            embed_message = self.embedding_handler.get_answer_from_embedding(best_matched_movie, user_query)
            return embed_message
        except:
            return ""

    #region SAPRQL factual questions

    def get_movie_entities_info_by_SPARQL(self, movie_list):

        res = []
        for m in movie_list:
            info = self._get_movie_info(m)
            res.append(info)
        
        if not res:
            print(f"No movie info found for movie list {movie_list}")
        return res

    def _get_movie_info(self, movie_name):
        query_template = '''
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?movie ?movieLabel ?predicateLabel ?object ?objectLabel WHERE {{
                # Find the movie entity based on an exact match for the label
                ?movie rdfs:label "{0}"@en .
                
                # Retrieve all predicates and objects related to the movie entity
                ?movie ?predicate ?object .

                FILTER(?predicate IN (
                      wdt:P31,   # instance of
                      wdt:P57,   # director
                      wdt:P162,  # producer
                      wdt:P364,  # original language
                      wdt:P272,  # production company
                      wdt:P58,   # screenwriter
                      wdt:P166,  # award received
                      wdt:P2047, # duration
                      wdt:P577 # release date
                  ))

                # Optionally retrieve labels for predicates and objects
                OPTIONAL {{ ?predicate rdfs:label ?predicateLabel . FILTER(LANG(?predicateLabel) = "en") }}
                OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "en") }}
                OPTIONAL {{ ?movie rdfs:label ?movieLabel . FILTER(LANG(?movieLabel) = "en") }}
            }}
            ORDER BY ?movie
        '''
        
        query = query_template.format(movie_name)
        result = self.graph.query(query)
        
        film_info = self.format_sparql_result(result, movie_name)

        return film_info
        
    def format_sparql_result(self, result, movie_name):
        """
        Convert the SPARQL query result into a dictionary with key-value pairs.
        """
        film_info = {}

        if not result:
            print(f"No information found for movie: {movie_name}")
            return list()
        
        def add_entity_key_value(entity_key, key, value):
            if entity_key not in film_info:
                film_info[entity_key] = defaultdict(list)
            film_info[entity_key][key].append(value)

        for row in result:
            entity_key, movie_label, label, obj, value = row
            if value is None:
                add_entity_key_value(str(entity_key.rsplit('/', 1)[-1]) + "--" + str(movie_label), str(label), str(obj))
            add_entity_key_value(str(entity_key.rsplit('/', 1)[-1]) + "--" + str(movie_label), str(label), str(value))

        return film_info
    
    #endregion SAPRQL factual questions

    def process_sparql_query(self, query) -> str:

        """
        Check if the query is valid, then execute the query and return the results.

        Returns:
            str: The results of the SPARQL query or error message if the query is not valid or error executing the query.
        
        Exceptions:
            Exception: If there is an error executing the query.
        """

        # Check if the query is valid
        is_valid, result = self._is_sparql_query_valid(query)

        if not is_valid:
            return f"The SPARQL query is not valid: {result}"

        query = result
        
        try:
            output = self._execute_query(query)

            # Send the output back to the chat room
            if output:
                return str(output)
            else:
                print(f"No results found for query: {query}")
                return "No results found. Would you like to input another query?"

        except Exception as e:
            print(f"Error executing query: {query}, the error message is {str(e)}")
            return f"Encountered error while executing query: {query}, the error message is {str(e)}"


    def is_sparql_query(self, str : str) -> bool:
        """
        Check if the string is a SELECT, ASK, CONSTRUCT, or DESCRIBE SPARQL query that we should process.
        """

        str = str.strip().upper()
        return any(keyword in str for keyword in ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE"))

    #region Private methods
    def _execute_query(self, query) -> list[str]:
        """Executes a sparql query and returns the results."""
        results = self.graph.query(query)

        # Collect results in a readable format
        output = []
        for row in results:
            for value in row:
                # Convert each value to string and append to output
                output.append(str(value))  # Convert to string for consistency
        
        return output
    
    def _is_sparql_query_valid(self, query) -> bool:
        """
        Check if the query is valid.

        Returns:
            bool: True if the query is valid, False otherwise.
            prepared query or error message: The prepared query if the query is valid, otherwise return error message .
        """

        try:
            # Attempt to prepare/parse the query
            prepared_query = prepareQuery(query)
            return True, prepared_query
        except Exception as e:
            return False, str(e)  # Return False and the error message
        
    def _get_graph_cache(self, graph_path, serialized_path):

        """
        Cache the RDF into binary file so that it will be faster to load next time
        Create the cache if it doesn't exist
        """

        # Check if serialized graph exists
        if os.path.exists(serialized_path):
            print("Loading serialized graph...")
            with open(serialized_path, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            # Load the graph from Turtle file if serialized graph doesn't exist
            print("Parsing KG file...")
            self.graph.parse(graph_path, format='turtle')
            
            # Serialize the graph for future use
            with open(serialized_path, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"Serialized graph saved to {serialized_path}")


    #endregion