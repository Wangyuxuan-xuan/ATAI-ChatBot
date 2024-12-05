import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from thefuzz import fuzz, process
import pickle

class NameEntityRecognizer:
    tuned_movie_bert_base_NER = "../Tune-BERT-NER/fine_tuned_BERT_base_uncased"

    def __init__(self):

        # Load the pre-trained BERT NER model from Hugging Face
        self.bert_base_NER = "dslim/bert-base-NER"
        self.bert_base_NER_tokenizer = AutoTokenizer.from_pretrained(self.bert_base_NER)
        self.bert_base_NER_model = AutoModelForTokenClassification.from_pretrained(self.bert_base_NER)
        self.bert_base_NER_pipeline = pipeline(
                    "ner",
                    model=self.bert_base_NER_model,
                    tokenizer=self.bert_base_NER_tokenizer,
                    aggregation_strategy="simple",
                    device="cuda"
                )        
        # Load self tuned BERT NER model
        self.tuned_movie_ner_pipeline = pipeline("ner", model=self.tuned_movie_bert_base_NER, tokenizer=self.tuned_movie_bert_base_NER, aggregation_strategy="simple", device="cuda")
        self._init_Dataset()

    def _init_Dataset(self):
        with open("../Dataset/MovieTitles.pickle", 'rb') as f:
            movie_titles = pickle.load(f)
        self.movie_title_set = set(movie_titles)

    def get_best_match_person(self, user_query: str) -> list:
        '''
        Extract the person name from user_query using bert_base_NER_pipeline.
        Concatenate the result, format the result to be the exact person name,
        and exclude unwanted characters.
        '''
        # Use the NER pipeline to get entities from the user query
        ner_results = self.bert_base_NER_pipeline(user_query)
        person_names = []
        current_name = ''
        for entity in ner_results:
            if entity['entity_group'] == 'PER':
                # Use the 'word' attribute to get the entity text
                word = entity['word']
                # Remove any leading/trailing punctuation and whitespace
                word = word.strip('.,!? ')
                # Check if the word starts with '##', indicating a continuation
                if word.startswith('##'):
                    # Remove '##' and concatenate without space
                    word = word[2:]
                    current_name += word
                else:
                    # If there's an existing name, append it to the list
                    if current_name:
                        person_names.append(current_name.strip())
                    # Start a new name
                    current_name = word
            else:
                # If we reach a non-PER entity, append the current name if it exists
                if current_name:
                    person_names.append(current_name.strip())
                    current_name = ''
        # Append any remaining name after the loop
        if current_name:
            person_names.append(current_name.strip())
        # Replace multiple spaces with a single space in each name
        person_names = [re.sub(r'\s+', ' ', name) for name in person_names]
        # Remove duplicates and return the list
        person_names = list(set(person_names))
        return person_names


    def get_matched_movies_list(self, user_query: str) -> list:
        ner_movies_arr = self.extract_movie_using_self_tuned_NER(user_query)

        matched_movies = []

        matched_movies = self.fuzzy_match_movie_with_movie_list(ner_movies_arr)

        res = []
        for m in matched_movies:
            res.append(m.lstrip())

        # Use bert_base_NER as fallback
        if not res:
            res = self.get_best_match_MISC_use_bert_base_NER(user_query)
        return res
    
    def get_best_match_MISC_use_bert_base_NER(user_query: str) -> list:
        '''
        Extract the person name from user_query using bert_base_NER_pipeline.
        Concatenate the result, format the result to be the exact person name,
        and exclude unwanted characters.
        '''
        # Use the NER pipeline to get entities from the user query
        ner_results = self.bert_base_NER_pipeline(user_query)
        movie_name = []
        current_name = ''
        for entity in ner_results:
            if entity['entity_group'] == 'MISC' or entity['entity_group'] == 'ORG':
                # Use the 'word' attribute to get the entity text
                word = entity['word']
                # Remove any leading/trailing punctuation and whitespace
                word = word.strip('.,!? ')
                # Check if the word starts with '##', indicating a continuation
                if word.startswith('##'):
                    # Remove '##' and concatenate without space
                    word = word[2:]
                    current_name += word
                else:
                    # If there's an existing name, append it to the list
                    if current_name:
                        movie_name.append(current_name.strip())
                    # Start a new name
                    current_name = word
            else:
                # If we reach a non-PER entity, append the current name if it exists
                if current_name:
                    movie_name.append(current_name.strip())
                    current_name = ''
        # Append any remaining name after the loop
        if current_name:
            movie_name.append(current_name.strip())
        # Replace multiple spaces with a single space in each name
        movie_name = [re.sub(r'\s+', ' ', name) for name in movie_name]
        # Remove duplicates and return the list
        movie_name = list(set(movie_name))
        return movie_name

    def fuzzy_match_movie_with_movie_list(self, ner_movies_arr:list):
        
        res = []
        for m in ner_movies_arr:
            best_match_list = self.find_topk_movie_match_from_movie_title_list(m, top_k=1)
            if best_match_list:
                res.append(best_match_list[0])

        return res

    def extract_movie_using_self_tuned_NER(self, sentence):
        ner_results = self.tuned_movie_ner_pipeline(sentence)

        movie_list = []

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
            movie = movie.replace('#', "")
            res.append(movie)
        
        return res
    
    def find_topk_movie_match_from_movie_title_list(self, ner_movie, top_k = 3):
        
        res = []
        # If no confident match was found, attempt secondary matching strategy
        # Try extracting the top 3 matches to see if a more suitable candidate exists
        extract_results = process.extract(ner_movie, self.movie_title_set, scorer=fuzz.ratio, limit=top_k)

        for match, score in extract_results:
            if not match:
                continue

            res.append(match)
            # print(f"{match}, {score}")

        return res

    def fuzzy_match_top_matched_movie_with_user_query(self, matched_movies, user_query) -> str:
        
        res = []

        extract_results = process.extract(user_query, matched_movies, scorer=fuzz.ratio, limit=2)

        for match, score in extract_results:
            if not match:
                continue
            
            if score < 80:
                continue

            res.append(match)
            # print(f"{match}, {score}")

        return res[0] if res else []

    def match_movie_list_with_user_query(self, matched_movies:list, user_query: str):

        def remove_non_alphanumeric(text):
            return ''.join(filter(str.isalnum, text))

        res = []
        # Hard constraint

        user_input_cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', user_query).lower().split()
        user_input_cleaned = remove_non_alphanumeric(user_input_cleaned)

        for m in matched_movies:
            m_cleaned = remove_non_alphanumeric(m)
            
            for word in user_input_cleaned:
                if m_cleaned.lower() == word:
                    res.append(m)

        
        return res
