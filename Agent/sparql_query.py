from rdflib import Graph

class SPARQLQueryExecutor:
    def __init__(self, dataset_path='./../Dataset/14_graph.nt'):
        self.graph = Graph()
        self.graph.parse(dataset_path, format='turtle')

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
        