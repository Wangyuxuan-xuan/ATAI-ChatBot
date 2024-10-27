from enum import Enum
import random
import threading
import time
from sparql_query import SPARQLQueryExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from movie_entity_extractor import MovieEntityExtractor
from transformers import StoppingCriteria, StoppingCriteriaList
from constants import SYNONYMS, SPARQL_RELATION_MAPPING, GREETING_SET, INITIAL_RESPONSES, PERIODIC_RESPONSES

class Intent(Enum):
    DIRECTOR = "director"
    RELEASE_DATE = "release_date"
    AWARD = "award"
    PRODUCTION_COMPANY = "production_company"
    LANGUAGE = "language"
    SCREENWRITER = "screenwriter"
    GENERAL_INFO = "general_info"

class response_generator:
    
    def __init__(self):
        self.sparql_executor = SPARQLQueryExecutor()

        # Initialize MovieEntityExtractor
        self.movie_entity_extractor = MovieEntityExtractor()

        # Load the RedPajama-INCITE-Chat-3B-v1 model
        model_name = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

        self.redpajama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.redpajama_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.redpajama_model = self.redpajama_model.to('cuda:0')
        self.redpajama_model.generation_config.pad_token_id = self.redpajama_tokenizer.pad_token_id

        self.room = None

    def set_room(self, room):
        """
        Set the current chatroom.

        Args:
            room: The chatroom object.
        """
        self.room = room

    def get_response(self, user_query: str) -> str:
        # Preprocess the user query to detect greetings
        processed_query = re.sub(r'[^a-zA-Z0-9 ]', '', user_query.lower().strip())
        if processed_query in GREETING_SET:
            return "Hello! I'm a movie chatbot, how can I help you today?"

        # Step 1: Perform NER
        matched_movies_list = self.movie_entity_extractor.get_matched_movies_list(user_query)
        print(f"matched movies: \n {matched_movies_list}")

        # Step 2: generate a SPARQL query beased on entities recognized
        movie_info = self.sparql_executor.get_movie_entities_info(matched_movies_list)

        # Step 3: Determine the intent of the question (If use hard-coded response generator)
        # intent = self.determine_intent(user_query)

        # Step 4: Format output using language model

        # response = self.generate_response_hardcoded(intent, movie_info)
        response = self.generate_response_using_redpajama(movie_info, user_query)
        return response

    #region LLM response generation

    def generate_response_using_redpajama(self, movie_info: dict, user_query: str) -> str:
        """
        Generate a response using redpajama based on the user query and the query result.
        """
        
        system_msg = '''
        You are a specialized movie chatbot to answer user queries in 1 short sentence

        INPORTANT: Provide exactly 1 short sentence as response, as short as possible

        Prioritize the provided information to formulate your response. 
        Exactly 1 short sentence as response, maximum 10 words.
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

        initial_message = random.choice(INITIAL_RESPONSES)
        if self.room:
            self.room.post_messages(initial_message)
        else:
            print(initial_message)

        # Function to post periodic intermediate responses
        def post_periodic_updates():
            start_time = time.time()
            while not hasattr(self, "generation_done"):
                if time.time() - start_time > 8:
                    periodic_message = random.choice(PERIODIC_RESPONSES)
                    if self.room:
                        self.room.post_messages(periodic_message)
                    else:
                        print(periodic_message)
                    start_time = time.time()
                    time.sleep(8)  # Post every 8 seconds

        # Start timer and thread for periodic updates
        update_thread = threading.Thread(target=post_periodic_updates)
        update_thread.start()

        # Generate response with stopping criteria
        outputs = None
        try:
            outputs = self.redpajama_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,    # Disable sampling to make the output deterministic
                temperature=1.0,    # No randomness in token selection
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                use_cache=True
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(1)
        
        # Mark generation as done
        self.generation_done = True
        update_thread.join()

        token = outputs.sequences[0, input_length:]
        output_str = self.redpajama_tokenizer.decode(token)

        response = self._format_output_by_LLM(output_str)
        return response

    def _format_output_by_LLM(self, output) -> str:
        # Remove the stop word from the output
        output_str = output.replace("<human>:", "").strip()

        if '.' in output_str:
            sentences = output_str.split('.')
            if len(sentences) > 1 and not output_str.endswith('.'):
                output_str = '. '.join(sentences[:-1]) + '.'
        
        return output_str


    #endregion LLM response generation
    
    #region Hard-coded response generation

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

    def generate_response_hardcoded(self, intent: Intent, movie_info: dict) -> str:
        response_parts = []
        for m_info in movie_info:
            response = self.generate_response_hardcoded_for_movie(intent, m_info)
            response_parts.append(response)
        return " \n".join(response_parts)

    def generate_response_hardcoded_for_movie(self, intent: Intent, structured_info: dict) -> str:
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

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False
