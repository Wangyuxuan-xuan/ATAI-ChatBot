from enum import Enum
from sparql_query import SPARQLQueryExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification, BitsAndBytesConfig, pipeline
import torch
from thefuzz import fuzz, process
import pickle
from transformers import StoppingCriteria, StoppingCriteriaList

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

        # Load the RedPajama-INCITE-Chat-3B-v1 model
        model_name = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

        self.redpajama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.redpajama_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.redpajama_model = self.redpajama_model.to('cuda:0')
        self.redpajama_model.generation_config.pad_token_id = self.redpajama_tokenizer.pad_token_id

    def _init_Dataset(self):
        with open("../Dataset/MovieTitles", 'rb') as f:
            movie_titles = pickle.load(f)
        self.movie_title_set = set(movie_titles)

    def get_response(self, user_query: str) -> str:

        # Step 1: Perform Named Entity Recognition (NER) on the message

        matched_movies_list = self.get_matched_movies_list(user_query)
        print(f"matched movies: \n {matched_movies_list}")

        # Step 2: generate a SPARQL query if entities are recognized
            
        movie_info = self.sparql_executor.get_movie_entities_info(matched_movies_list)

        # Step 3: Determine the intent of the question
        intent = self.determine_intent(user_query)

        # Step 4: Format output using language model

        response = self.generate_response_using_redpajama(movie_info, user_query)
        return response

    #region LLM response generation

    def generate_response_using_redpajama(self, movie_info: dict, user_query: str) -> str:
        """
        Generate a response using redpajama based on the user query and the query result.
        """
        
        system_msg = '''
        You are a specialized movie chatbot. 

        <Requirements>
        INPORTANT: Provide exactly 1 short sentence as response, as short as possible (within 10 words)
        <Requirements>

        Prioritize the provided information to formulate your response. 

        The questions you recieved are factual questions, for yes or no questions just respond in a few words.
        '''

        prompt_info = f"<system>: \"{system_msg}\"\n"
        prompt_info += f"<User query>: \"{user_query}\"\n"
        prompt_info += f"<data>: \"{movie_info}\"\n"

        # Format the prompt with the movie information and user query
        prompt = f"<human>: {prompt_info}\n<bot>: "

        # Tokenize and prepare the input for the model
        inputs = self.redpajama_tokenizer(prompt, return_tensors="pt").to("cuda")

        input_length = inputs.input_ids.shape[1]

            # Define stopping criteria to stop after generating the bot response
        stop_words = ["<human>:"]
        stop_words_ids = [
            self.redpajama_tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in stop_words
        ]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # Generate response with stopping criteria
        outputs = self.redpajama_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,    # Disable sampling to make the output deterministic
            temperature=1.0,    # No randomness in token selection
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True
        )


        token = outputs.sequences[0, input_length:]
        output_str = self.redpajama_tokenizer.decode(token)

        # Remove the stop word from the output
        output_str = output_str.replace("<human>:", "").strip()

        if '.' in output_str:
            sentences = output_str.split('.')
            if len(sentences) > 1 and not output_str.endswith('.'):
                output_str = '. '.join(sentences[:-1]) + '.'
        
        return output_str

    #endregion LLM response generation
    
    #region hard-coded response generation

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

    def generate_response_hardcoded(self, intent: Intent, structured_info: dict) -> str:
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

    #endregion

    #region NER

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
   
    #endregion NER

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False
