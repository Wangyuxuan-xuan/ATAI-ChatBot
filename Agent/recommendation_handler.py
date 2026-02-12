import warnings
from sklearn.metrics import pairwise_distances
from graph_processor import GraphProcessor
from sklearn.cluster import KMeans
import numpy as np

class RecommendationHandler:

    def __init__(self, graph_processor: GraphProcessor):
        self.graph_processor = graph_processor
        # Suppress specific warning category from sklearn
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

        # Define a whitelist of meaningful relations


        self.relation_whitelist = [
            'director', 'characters', 'screenwriter', 'award received', 'film editor', 'genre', 'nominated for', 'published in', 'award received', 'part of the series',
            'production designer', 'production company', 'inspired by'
            ]


    def recommend_movies(self, liked_movies: list):

        top_k = 3

        # Extract embedding-based recommendations
        recommendations = []
        
        similar_movies = self.graph_processor.embedding_handler.get_similar_movies(liked_movies, 10)

        features = self.get_top_k_features(liked_movies, top_k = 3)
        
        # Remove duplicates and return recommendations
        for m in similar_movies:
            if m in liked_movies:
                continue
            
            if m in recommendations:
                continue
            
            recommendations.append(m)

        recommendations = recommendations[:top_k]  # Limit to top_k results
        return features, recommendations

    def recommend_movie_based_on_director_or_actor(self, person_name_list):
        res = []

        for p in person_name_list:
            recommendation = self._recommend_movie_based_on_director_or_actor(p)
            res.append(recommendation)
        
        return res

    def _recommend_movie_based_on_director_or_actor(self, person_name):
        # Convert person label to entity URI
        if person_name not in self.graph_processor.embedding_handler.lbl2ent:
            # Person not found in the KG embedding dictionary
            return []

        person_uri = self.graph_processor.embedding_handler.lbl2ent[person_name]

        # SPARQL Query:
        # We look for movies (wdt:P31 wd:Q11424) for which the given person is either a director (wdt:P57)
        # or an actor (wdt:P161).
        query = f'''
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?movie ?movieLabel WHERE {{
              ?movie wdt:P31 wd:Q11424.
              {{
                ?movie wdt:P57 <{person_uri}>  # Director
              }} UNION {{
                ?movie wdt:P161 <{person_uri}> # Actor
              }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
        '''

        results = self.graph_processor.graph.query(query)

        # Extract movie labels
        movie_results = []
        for row in results:
            movie_uri = row[0]
            movie_label = str(row[1])
            movie_results.append(movie_label)

        # Return top 3 if multiple found
        return movie_results[:3]

    def get_top_k_features(self, liked_movies: list, top_k: int = 3) -> list:
        """
        Recommend movies based on all features of liked movies using K-means clustering.
        Args:
            liked_movies (list): List of liked movie names.
            top_k (int): Number of recommendations to return.
        Returns:
            list: A list of recommended movies.
        """
        # Get embeddings for all features of liked movies
        feature_embeddings = self.get_all_feature_embeddings(liked_movies)

        if not feature_embeddings:
            return []

        # Apply K-means clustering
        n_clusters = min(len(set(map(tuple, feature_embeddings))), max(1, len(feature_embeddings) // 2))
        cluster = KMeans(n_clusters=n_clusters, n_init="auto")
        cluster.fit(np.array(feature_embeddings))

        # Get top clusters and recommend movies based on centroids
        cluster_counts = np.bincount(cluster.labels_)
        cluster_idx = np.argsort(cluster_counts)[-top_k:]

        centroids = cluster.cluster_centers_[cluster_idx]
        dist = pairwise_distances(centroids, self.graph_processor.embedding_handler.ent_embeds)

        # select features based on the most K-th significant centroids 
        closest_rel_idx = dist.argsort()[:,0]
        # closest_rel_lbl = [id2ent[i] for i in closest_rel_idx]
        closest_rel_uri = [self.graph_processor.embedding_handler.id2ent[i] for i in closest_rel_idx]
        closest_rel_lbl = [self.graph_processor.embedding_handler.ent2lbl[i] for i in closest_rel_uri]

        return closest_rel_lbl


    def get_all_feature_embeddings(self, movie_names: list) -> list:
        """
        Retrieve embeddings for entities related to given movies, filtered by meaningful relations.
        Args:
            movie_names (list): List of movie names.
        Returns:
            list: A list of embeddings for filtered entities across the movies.
        """
        all_feature_embeddings = []
        feature_entities = []

        for movie_name in movie_names:
            if movie_name not in self.graph_processor.embedding_handler.lbl2ent:
                continue

            movie_uri = self.graph_processor.embedding_handler.lbl2ent[movie_name]
            if movie_uri not in self.graph_processor.embedding_handler.ent2id:
                continue

            # Define meaningful predicates for filtering
            predicate_filter = '''
                wdt:P31,   # instance of
                wdt:P57,   # director
                wdt:P162,  # producer
                wdt:P272,  # production company
                wdt:P58,   # screenwriter
                wdt:P166,  # award received
                wdt:P577,  # release date
                wdt:P136   # genre
            '''

            # Query the KG for entities related by meaningful predicates
            query_template = '''
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                
                SELECT ?object ?objectLabel WHERE {{
                    <{0}> ?predicate ?object .
                    FILTER(?predicate IN ({1}))
                    OPTIONAL {{ ?object rdfs:label ?objectLabel . FILTER(LANG(?objectLabel) = "en") }}
                }}
            '''
            query = query_template.format(movie_uri, predicate_filter)
            entities = self.graph_processor.graph.query(query)

            for entity in entities:
                entity_uri, entity_label = entity[0], str(entity[1])
                if entity_uri in self.graph_processor.embedding_handler.ent2id:
                    feature_entities.append(entity_label)

                    # Add entity embedding
                    entity_id = self.graph_processor.embedding_handler.ent2id[entity_uri]
                    entity_embed = self.graph_processor.embedding_handler.ent_embeds[entity_id]
                    all_feature_embeddings.append(entity_embed)

        return all_feature_embeddings
    

    # def _recommend_by_features_by_KG(self, features: list, top_k: int = 1) -> dict:
    #     """
    #     Recommend one movie for each extracted feature using Knowledge Graph.
    #     Args:
    #         features (list): List of feature entities (e.g., directors, production companies).
    #         top_k (int): Number of recommended movies per feature (default is 1).
    #     Returns:
    #         dict: A dictionary with features as keys and recommended movie names as values.
    #     """
    #     recommendations = {}

    #     # Iterate over each feature to find related movies in the KG
    #     for feature in features:
    #         if feature not in self.graph_processor.embedding_handler.lbl2ent:
    #             continue

    #         feature_uri = self.graph_processor.embedding_handler.lbl2ent[feature]

    #         # Query KG for movies related to the feature
    #         query_template = '''
    #             PREFIX wd: <http://www.wikidata.org/entity/>
    #             PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    #             PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    #             SELECT ?movie ?movieLabel WHERE {{
    #                 ?movie ?predicate <{0}> .
    #                 ?movie wdt:P31 wd:Q11424 .  # Ensure it is an instance of "film"
    #                 OPTIONAL {{ ?movie rdfs:label ?movieLabel . FILTER(LANG(?movieLabel) = "en") }}
    #             }}
    #             LIMIT {1}
    #         '''
    #         query = query_template.format(feature_uri, top_k)
    #         results = self.graph_processor.graph.query(query)

    #         # Collect movie recommendations for this feature
    #         feature_recommendations = []
    #         for row in results:
    #             movie_uri, movie_label = row
    #             if movie_label:
    #                 feature_recommendations.append(str(movie_label))

    #         if feature_recommendations:
    #             recommendations[feature] = feature_recommendations

    #     return recommendations
