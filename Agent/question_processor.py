from sparql_query import SPARQLQueryExecutor

class question_processor:
    def __init__(self):
        self.sparql_executor = SPARQLQueryExecutor()


    def get_response(self, message: str) -> str:
        # Check if it is a SPARQL query
        if not self.sparql_executor.is_sparql_query(message):
            return "I can only process SPARQL queries. Please enter a SPARQL query only."
        else:
            return self.sparql_executor.process_sparql_query(message)


    