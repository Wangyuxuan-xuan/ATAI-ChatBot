import rdflib

class SPARQLProcessor:
    def __init__(self, graph_path):
        self.graph = rdflib.Graph()
        self.graph.parse(graph_path, format='turtle')

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