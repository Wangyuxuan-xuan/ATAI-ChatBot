from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from thefuzz import fuzz, process
import pickle

class MovieEntityExtractor:
    tuned_movie_bert_base_NER = "../Tune-BERT-NER/Tuned_BERT_NER_movie-60000"

    def __init__(self):

        # # Load the pre-trained BERT NER model from Hugging Face
        # self.bert_base_NER = "dslim/bert-base-NER"
        # self.bert_base_NER_tokenizer = AutoTokenizer.from_pretrained(self.bert_base_NER)
        # self.bert_base_NER_model = AutoModelForTokenClassification.from_pretrained(self.bert_base_NER)
        # self.bert_base_NER_pipeline = pipeline("ner", model=self.bert_base_NER_model, tokenizer=self.bert_base_NER_tokenizer)
        
        # Load self tuned BERT NER model
        self.tuned_movie_ner_pipeline = pipeline("ner", model=self.tuned_movie_bert_base_NER, tokenizer=self.tuned_movie_bert_base_NER, aggregation_strategy="simple", device="cuda")
        self._init_Dataset()

    def _init_Dataset(self):
        with open("../Dataset/MovieTitles", 'rb') as f:
            movie_titles = pickle.load(f)
        self.movie_title_set = set(movie_titles)

    def get_matched_movies_list(self, user_query: str) -> list:
        ner_movies_arr = self.extract_movie_using_self_tuned_NER(user_query)

        matched_movies = []
        for m in ner_movies_arr:
            fuzzy_matched = self.find_top3_movie_match(m)
            matched_movies.extend(fuzzy_matched)

        matched_movies = self.filter_matched_movies(matched_movies, user_query)
        return matched_movies

    def extract_movie_using_self_tuned_NER(self, sentence):
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
    
    def find_top3_movie_match(self, ner_movie):
        
        res = []
        # If no confident match was found, attempt secondary matching strategy
        # Try extracting the top 3 matches to see if a more suitable candidate exists
        extract_results = process.extract(ner_movie, self.movie_title_set, scorer=fuzz.ratio, limit=3)

        for match, score in extract_results:
            if not match:
                continue

            res.append(match)
            print(f"{match}, {score}")

        return res

    def filter_matched_movies(self, matched_movies:list, user_input):

        def remove_non_alphanumeric(text):
            return ''.join(filter(str.isalnum, text))

        res = []
        for m in matched_movies:
            m_cleaned = remove_non_alphanumeric(m)
            user_input_cleaned = remove_non_alphanumeric(user_input)
            if m_cleaned.lower() in user_input_cleaned.lower():
                res.append(m)

        return res
