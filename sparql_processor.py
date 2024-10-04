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

    def process_query(self, query):
        try:
            results = self.graph.query(query)
            return [str(row) for row in results]
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def generate_response(self, message):
        if message.lower().startswith("sparql:"):
            query = message[7:].strip()
            results = self.process_query(query)
            return f"Query results: {results}"
        else:
            return "I can only process SPARQL queries. Please start your message with 'SPARQL:' followed by your query."
