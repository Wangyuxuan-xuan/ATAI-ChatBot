import rdflib
import pickle
import os

class SPARQLProcessor:


    CACHE_GRAPH_PATH = './Dataset/graph.pkl'
    
    def __init__(self, graph_path, serialized_path=CACHE_GRAPH_PATH):
        self.graph = rdflib.Graph()
        self.get_graph_cache(graph_path, serialized_path)

    def get_graph_cache(self, graph_path, serialized_path):
        # Check if serialized graph exists
        if os.path.exists(serialized_path):
            print("Loading serialized graph...")
            with open(serialized_path, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            # Load the graph from Turtle file if serialized graph doesn't exist
            print("Parsing Turtle file...")
            self.graph.parse(graph_path, format='turtle')
            
            # Serialize the graph for future use
            with open(serialized_path, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"Serialized graph saved to {serialized_path}")


    def generate_response(self, message):

        if "SELECT" not in message.upper():
            return "I can only process SPARQL queries. Please enter a SPARQL query only."

        try:
            queryResult = self.process_query(message)
            return self.format_results(queryResult)

        except Exception as e:
            return f"Error executing query: {str(e)}"

        

    def process_query(self, query):
        results = self.graph.query(query)
        headers = results.vars
        rows = []
        
        # Extract each row as a dictionary of variable-value pairs
        for row in results:
            row_dict = {str(var): str(row[var]) for var in headers}
            rows.append(row_dict)
        
        return headers, rows        

    def format_results(self, queryResult):

        headers, results = queryResult

        if isinstance(headers, str):
            return headers  # Return error message if process_query returned an error string
        
        if not results:
            return "No results found for the query."
        
        # Format results as a table-like output
        output = "Query Results:\n"
        output += "\t".join(str(h) for h in headers) + "\n"  # Add headers
        output += "-" * 40 + "\n"            # Add separator
        for row in results:
            output += "\t".join(row.get(str(var), '') for var in headers) + "\n"
    
        return output