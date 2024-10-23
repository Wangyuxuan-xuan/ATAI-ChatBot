from enum import Enum
from sparql_query import SPARQLQueryExecutor
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import spacy
import torch
from rapidfuzz import process
import pickle

class Intent(Enum):
    DIRECTOR = "director"
    RELEASE_DATE = "release_date"
    AWARD = "award"
    PRODUCTION_COMPANY = "production_company"
    LANGUAGE = "language"
    SCREENWRITER = "screenwriter"
    GENERAL_INFO = "general_info"

SYNONYMS = {
    "director": ["director", "directed", "directs", "direct"],
    "release_date": ["release date", "released"],
    "award": ["award", "oscar", "prize"],
    "production_company": ["production company", "produced"],
    "language": ["language", "original language"],
    "screenwriter": ["screenwriter", "writer"]
}

SPARQL_RELATION_MAPPING = {
    "director": "director",
    "release_date": "publication date",
    "award": "award received",
    "production_company": "production company",
    "language": "original language of film or TV show",
    "screenwriter": "screenwriter"
}

class response_generator:

    bert_base_NER = "dslim/bert-base-NER"
    tuned_movie_bert_base_NER = "../Tune-BERT-NER/Tuned_BERT_NER_movie-60000"


    
    def __init__(self):
        self.sparql_executor = SPARQLQueryExecutor()
        # Load the pre-trained BERT NER model from Hugging Face
        
        self.bert_base_NER_tokenizer = AutoTokenizer.from_pretrained(self.bert_base_NER)
        self.bert_base_NER_model = AutoModelForTokenClassification.from_pretrained(self.bert_base_NER)
        self.bert_base_NER_pipeline = pipeline("ner", model=self.bert_base_NER_model, tokenizer=self.bert_base_NER_tokenizer)
        
        # Load self tuned movie NER model
        
        self.tuned_movie_ner_pipeline = pipeline("ner", model=self.tuned_movie_bert_base_NER, tokenizer=self.tuned_movie_bert_base_NER, aggregation_strategy="simple", device="cuda")
        self._init_Dataset()
        
    def _init_Dataset(self):
        with open("../Dataset/MovieTitles", 'rb') as f:
            movie_titles = pickle.load(f)
        self.movie_title_set = set(movie_titles)

    def get_response(self, message: str) -> str:

        # Step 1: Perform Named Entity Recognition (NER) on the message
        ner_person = self.extract_person(message)
        ner_movies = self.extract_movie(message)

        e_movies = self.do_fuzz_match(ner_movies)

        # Step 2: generate a SPARQL query if entities are recognized
        if e_movies :
            
            movie_info = self.sparql_executor.get_entities_info(e_movies, ner_person)

            if not movie_info or len(movie_info) == 0:
                # Default response if no valid entities or intent are found
                return "I'm not sure how to answer that. Could you please rephrase or provide more details?"

            # Step 3: Determine the intent of the question
            intent = self.determine_intent(message)

            # Step 4: Extract the relevant information based on intent

            response_parts = []
            for m_info in movie_info:
                response = self.generate_response(intent, m_info)
                response_parts.append(response)
            return " \n".join(response_parts)


    def preprocess_message(self, message: str) -> str:
        """
        Preprocess the message to replace synonyms for more accurate intent recognition.
        """
        message = message.lower()
        for key, synonyms in SYNONYMS.items():
            for synonym in synonyms:
                if synonym in message:
                    message = message.replace(synonym, key)
        return message

    def determine_intent(self, message: str) -> Intent:
        """
        Determine the user's intent based on keywords in the message.
        """
        message = self.preprocess_message(message)

        for intent_key in SYNONYMS.keys():
            if intent_key in message:
                return Intent(intent_key)
        return Intent.GENERAL_INFO

    def generate_response(self, intent: Intent, structured_info: dict) -> str:
        """
        Generate a response based on the user's intent and the structured SPARQL query result.
        """

        if not structured_info or len(structured_info) == 0:
            # No data generated from SPARQL
            return ""
 

        response_parts = []
        sparql_relation = SPARQL_RELATION_MAPPING.get(intent.value)


        if len(structured_info) > 1:
            self._handle_multiple_match(structured_info, response_parts, sparql_relation)
        else:
            for movie, details in structured_info.items():
                movie_name = movie.split('--')[1]
                if sparql_relation and sparql_relation in details:
                    values = [v for v in details[sparql_relation] if v != "None"]
                    if values:
                        response_parts.append(f"The {intent.value.replace('_', ' ')} of {movie_name} is {', '.join(values)}.")
                else:
                    general_info = [f"{key}: {', '.join([v for v in values if v != 'None'])}" for key, values in details.items()]
                    response_parts.append(f"Here is some information about {movie_name}: {'; '.join(general_info)}.")

        return " \n".join(response_parts)

    def _handle_multiple_match(self, structured_info, response_parts, sparql_relation):

        response_parts.append(f"There are multiple instances of the entity you mentioned:")

        for movie, details in structured_info.items():
                movie_name = movie.split('--')[1]
                instance_type = details.get("instance of", ["Unknown"])[0]
                if sparql_relation and sparql_relation in details:
                    values = [v for v in details[sparql_relation] if v != "None"]
                    if values:
                        response_parts.append(f"The {instance_type} of {movie_name} is released on {', '.join(values)}.")
                else:
                    general_info = [f"{key}: {', '.join([v for v in values if v != 'None'])}" for key, values in details.items()]
                    response_parts.append(f"Here is some information about {movie_name} ({instance_type}): {'; '.join(general_info)}.")

    def extract_movie(self, sentence):
        ner_results = self.tuned_movie_ner_pipeline(sentence)

        movie_list = []
        print(f"\nInput  Sentence: \"{sentence}\"")

        if not ner_results:
            print("No entities found.")
        else:
            cur_movie = ""
            for entity in ner_results:
                label = entity["entity_group"]
                word = entity["word"]
                # Start of movie title
                if label == "LABEL_1":
                    # Append the previous movie to list
                    if cur_movie:
                        movie_list.append(cur_movie)
                    
                    cur_movie = word
                elif label == "LABEL_2":
                    cur_movie += word
            
            if cur_movie:
                movie_list.append(cur_movie) 

            movie_list = self._cleanup_movies_list(movie_list)
            
            return movie_list
    def _cleanup_movies_list(self, movie_list) -> list:
        res = []
        # Clean up
        for movie in movie_list:
            movie = movie.replace('"', "")
            res.append(movie)
        
        return res

    def extract_person(self, sentence):
        # Load pre-trained model and tokenizer

        # Tokenize the sentence and obtain model outputs
        inputs = self.bert_base_NER_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = self.bert_base_NER_model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0].tolist()

        # Convert token and label IDs to strings
        tokens = self.bert_base_NER_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.bert_base_NER_model.config.id2label[label_id] for label_id in predictions]

        # Collect persons and their labels
        p_entities = []
        entity = ""
        current_label = None  # Keep track of the current entity label
        for token, label in zip(tokens, labels):
            if label.startswith("B") or label.startswith("I"):  # Beginning or Inside of an entity
                if token.startswith("##"):  # Continuation of a word
                    entity += token[2:]  # Remove subword prefix
                else:  # New word
                    entity += " " + token  # Add space before new word
                current_label = label  # Update current entity label
            elif entity:  # Outside of an entity, but entity string is non-empty
                entity = entity.strip()  # Remove trailing space
                if current_label == "I-PER":
                    p_entities.append(entity)
                entity = ""  # Reset entity string
                current_label = None  # Reset current entity label

        # If sentence ends with an entity, append it to the appropriate list
        if entity:
            entity = entity.strip()
            if current_label == "I-PER":
                p_entities.append(entity)

        return p_entities

    
    def do_fuzz_match(self, ner_entites) -> list:
        res = []
        for e in ner_entites:
            best_match = self.find_best_match(e)
            if best_match:
                res.append(best_match)
        return res
    
    def find_best_match(self, ner_movie):

        best_match, score, index = process.extractOne(ner_movie, self.movie_title_set)
        # tuple containing the best matching movie title and a score
        if score > 80:  # Adjust the threshold as needed
            return best_match
        return None
