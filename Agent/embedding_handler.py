import csv
from enum import Enum
import numpy as np

import rdflib
from sklearn.metrics import pairwise_distances

from constants import EMBEDDING_REL_MAPPING

class Relation(Enum):
    DIRECTOR = "director"
    PUBLICATION_DATE = "publication date"
    EXECUTIVE_PRODUCER = "executive producer"
    SCREENWRITER = "screenwriter"
    FILM_EDITOR = "film editor"
    BOX_OFFICE = "box office"
    COST = "cost"
    NOMINATED_FOR = "nominated for"
    PRODUCTION_COMPANY = "production company"
    COUNTRY_OF_ORIGIN = "country of origin"
    CAST_MEMBER = "cast member"
    GENRE = "genre"
    
class EmbeddingHandler:

    def __init__(self, graph, ent2lbl):
        print("Loading Embedding data")
        self.graph = graph
        self.ent2lbl = ent2lbl
        #Load entity and relation embeddings
        self.ent_embeds = np.load("./../Dataset/Embeddings/entity_embeds.npy")
        self.rel_embeds = np.load("./../Dataset/Embeddings/relation_embeds.npy")
        
        # Load entity and relation dictionaries from .del files
        with open("./../Dataset/Embeddings/entity_ids.del", 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            # Create reverse dictionaries
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open("./../Dataset/Embeddings/relation_ids.del", 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

        # self.ent2lbl = {ent: str(lbl) for ent, lbl in graph.subject_objects(rdflib.RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}
        self.rel2lbl = {k:v for k, v in self.ent2lbl.items() if self._is_relation(k)}
        self.lbl2rel = {lbl: ent for ent, lbl in self.rel2lbl.items()}

        print("Loaded Embedding data")

    def get_answer_from_embedding(self, movie_name, user_query) -> str:
        """
        Use embeddings to retrieve the information related to a movie.
        """
        if not movie_name:
            print("No movie name provided to perform embedding lookup.")
            return ""

        print(f"------ Embedding search ------")
        # Find the entity URI corresponding to the movie name label
        if movie_name not in self.lbl2ent:
            print(f"The movie '{movie_name}' was not found in the embedding space.")
            return ""

        movie_uri = self.lbl2ent[movie_name]

        # Check if the movie URI is present in the entity dictionary
        if movie_uri not in self.ent2id:
            print(f"The movie '{movie_name}' does not have an associated embedding in the space.")
            return ""

        # Retrieve the entity embedding for the movie
        movie_id = self.ent2id[movie_uri]
        movie_embed = self.ent_embeds[movie_id].reshape(1, -1)

        intent = self.get_embedding_relation(user_query)
        if not intent:
            return ""
        relation_label = intent.value 

        relation_uri = self.lbl2rel[relation_label]

        if relation_uri not in self.rel2id:
            print(f"The requested relation '{relation_label}' was not found in the embedding space.")
            return ""

        # Retrieve the relation embedding
        relation_id = self.rel2id[relation_uri]
        relation_embed = self.rel_embeds[relation_id].reshape(1, -1)

        # Compute the combined embedding (movie + relation)
        combined_embed = movie_embed + relation_embed

        dist = pairwise_distances(combined_embed, self.ent_embeds).flatten()
        most_likely_idx = dist.argsort()[0]
        best_match_entity = self.id2ent[most_likely_idx]

        best_match_label = self.ent2lbl.get(best_match_entity, str(best_match_entity))

        # Return the result
        response = f"Answer suggested by embedding: The {relation_label.replace('_', ' ')} of {movie_name} is {best_match_label}."
        return response


    def get_similar_movies(self, movie_names: list, top_k: int = 10) -> list:
        """
        Retrieve the top K movies similar to the average embedding of provided movie names.
        Args:
            movie_names (list): List of movie names for which to find similar movies.
            top_k (int): Number of similar movies to retrieve.
        Returns:
            list: A list of movie names that are closest to the average embedding.
        """
        movie_embeds = []

        # Retrieve the embedding for each movie
        for movie_name in movie_names:
            if movie_name not in self.lbl2ent:
                print(f"The movie '{movie_name}' is not found in the embedding space.")
                continue
            movie_uri = self.lbl2ent[movie_name]
            if movie_uri not in self.ent2id:
                print(f"The movie '{movie_name}' does not have an associated embedding.")
                continue
            movie_id = self.ent2id[movie_uri]
            movie_embeds.append(self.ent_embeds[movie_id])

        # If no valid embeddings are found, return an empty list
        if not movie_embeds:
            return []

        # Calculate the mean embedding
        mean_embedding = np.mean(movie_embeds, axis=0).reshape(1, -1)

        # Calculate distances to all other movies based on the mean embedding
        dist = pairwise_distances(mean_embedding, self.ent_embeds).flatten()
        similar_movie_ids = dist.argsort()[:top_k]  # Get the top_k closest movies

        # Map the closest IDs back to movie names
        similar_movies = [self.ent2lbl[self.id2ent[idx]] for idx in similar_movie_ids if idx in self.id2ent]
        
        return similar_movies
    

    def get_feature_embeddings(self, movie_names: list, relation_label: str = "genre") -> list:
        """
        Retrieve embeddings of a specified feature (e.g., genre) for given movies.
        Args:
            movie_names (list): List of movie names.
            feature (str): Feature to retrieve embeddings for.
        Returns:
            list: A list of embeddings for the specified feature across the movies.
        """
        relation_embeddings = []

        # Ensure the feature is valid
        relation_uri = self.lbl2rel[relation_label]

        if relation_uri not in self.rel2id:
            print(f"The requested relation '{relation_label}' was not found in the embedding space.")
            return ""

        # Retrieve the relation embedding
        relation_id = self.rel2id[relation_uri]
        relation_embed = self.rel_embeds[relation_id].reshape(1, -1)

        for movie_name in movie_names:
            if movie_name not in self.lbl2ent:
                continue

            movie_uri = self.lbl2ent[movie_name]
            if movie_uri not in self.ent2id:
                continue

            movie_id = self.ent2id[movie_uri]
            movie_embed = self.ent_embeds[movie_id].reshape(1, -1)
            combined_embedding = movie_embed + relation_embed
            relation_embeddings.append(combined_embedding.flatten())

        return relation_embeddings

    
    def get_embedding_relation(self, message: str) -> Relation:
        """
        Determine the user's intent based on keywords in the message.
        """
        message = self.preprocess_message(message)

        for intent_key in EMBEDDING_REL_MAPPING.keys():
            if intent_key in message:
                return Relation(intent_key)
        return None
    
    def preprocess_message(self, message: str) -> str:
        """
        Preprocess the message to replace synonyms for more accurate intent recognition.
        """
        message = message.lower()
        for key, synonyms in EMBEDDING_REL_MAPPING.items():
            for synonym in synonyms:
                if synonym in message:
                    message = message.replace(synonym, key)
        return message
    
    def _is_relation(self, URI):
        label = self._get_label(URI)
        return label[0] == 'P'

    def _get_label(self, URI):
        return str(URI).split('/')[-1]