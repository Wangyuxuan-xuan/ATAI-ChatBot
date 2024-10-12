import pickle
import os
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

class SPARQLQueryExecutor:
    KG_GRAPH_PATH = './../Dataset/14_graph.nt'
    CACHE_GRAPH_PATH = './../Dataset/graph.pkl'
    
    def __init__(self, dataset_path= KG_GRAPH_PATH):
        self.graph = Graph()
        self._get_graph_cache(dataset_path, self.CACHE_GRAPH_PATH)
    
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
            # After cache 30-40 s
            # Before cache 


    #endregion