import pickle
import os
from rdflib import Graph

class SPARQLQueryExecutor:
    KG_GRAPH_PATH = './../Dataset/14_graph.nt'
    CACHE_GRAPH_PATH = './../Dataset/graph.pkl'
    
    def __init__(self, dataset_path= KG_GRAPH_PATH):
        self.graph = Graph()
        self.get_graph_cache(dataset_path, self.CACHE_GRAPH_PATH)

    def execute_query(self, query):
        """Executes a SPARQL query and returns the results."""
        results = self.graph.query(query)

        # Collect results in a readable format
        output = []
        for row in results:
            for value in row:
                # Convert each value to string and append to output
                output.append(str(value))  # Convert to string for consistency
        
        return output
        
    def get_graph_cache(self, graph_path, serialized_path):
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