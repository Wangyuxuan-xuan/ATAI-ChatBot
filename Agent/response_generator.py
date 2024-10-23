from sparql_query import SPARQLQueryExecutor
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import spacy
import torch

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

    def get_response(self, message: str) -> str:

        # Step 1: Perform Named Entity Recognition (NER) on the message
        e_person = self.extract_person(message)
        e_movies = self.extract_movie(message)
        # Step 2: Determine intent and generate a SPARQL query if entities are recognized
        if e_movies :
            
            info = self.sparql_executor.get_entities_info(e_movies, e_person)

            if not info:
                # Default response if no valid entities or intent are found
                return "I'm not sure how to answer that. Could you please rephrase or provide more details?"

            return str(info)

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

