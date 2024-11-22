from enum import Enum
from recommendation_handler import RecommendationHandler
from graph_processor import GraphProcessor
from transformers import pipeline
import torch
import re
import random
from movie_entity_extractor import MovieEntityExtractor
from constants import SYNONYMS, SPARQL_RELATION_MAPPING, GREETING_SET, INITIAL_RESPONSES, PERIODIC_RESPONSES

class Intent(Enum):
    DIRECTOR = "director"
    RELEASE_DATE = "release_date"
    AWARD = "award"
    PRODUCTION_COMPANY = "production_company"
    LANGUAGE = "language"
    SCREENWRITER = "screenwriter"
    GENERAL_INFO = "general_info"

class QuestionType(Enum):
    FACTUAL = "Factual"
    RECOMMENDATION = "Recommendation"
    MULTIMEDIA = "Multimedia"
    UNRELATED = "Unrelated"

class response_generator:
    
    question_classifier_path = "../QuestionClassifier/svm_question_classifier.pth"

    def __init__(self):
        self.graph_processor = GraphProcessor()

        # Initialize MovieEntityExtractor
        self.movie_entity_extractor = MovieEntityExtractor()

        self.recommendation_handler = RecommendationHandler(self.graph_processor)

        # Initialize Llama-3.2-1B-Instruct
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        access_token = "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"

        self.llama_pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  
            token =access_token  
        )

        # Load the question classifier model and vectorizer
        checkpoint = torch.load(self.question_classifier_path, weights_only = False)
        self.question_classifier = checkpoint['svm_model']
        self.question_classifier_vectorizer = checkpoint['vectorizer']

    def get_response(self, user_query: str) -> str:
        # Preprocess the user query to detect greetings
        processed_query = re.sub(r'[^a-zA-Z0-9 ]', '', user_query.lower().strip())
        if processed_query in GREETING_SET:
            return "Hello! I'm a movie chatbot, how can I help you today?"

        # Step 1: Perform NER
        matched_movies_list = self.movie_entity_extractor.get_matched_movies_list(user_query)
        print(f"matched movies: \n {matched_movies_list}")

        question_type: QuestionType = self._get_question_type(user_query)

        response = ""

        if question_type == QuestionType.FACTUAL:
            response = self._answer_factual_questions(user_query, matched_movies_list)
        elif question_type == QuestionType.RECOMMENDATION:
            response = self._answer_recommendation_questions(user_query, matched_movies_list)
        elif question_type == QuestionType.MULTIMEDIA:
            response = "I am sorry, multimedia is not supported yet"
        else:
            response = self._handle_unrelated_questions(user_query)
        
        return response
    
    def _handle_unrelated_questions(self, user_query: str):
        return "Appoligze that I have knowledge of movie related questions only."

    def _get_question_type(self, user_query) -> QuestionType:

        fall_back_type = QuestionType.FACTUAL

        # Use the loaded model for inference
        new_questions = [user_query]
        new_questions_tfidf = self.question_classifier_vectorizer.transform(new_questions)
        predictions = self.question_classifier.predict(new_questions_tfidf)
        
        if not predictions:
            return fall_back_type
        
        type = predictions[0]
        
        match type:
            case "Factual": return QuestionType.FACTUAL
            case "Recommendation": return QuestionType.RECOMMENDATION
            case "Multimedia": return QuestionType.MULTIMEDIA
            case "Unrelated": return QuestionType.UNRELATED


    def _answer_factual_questions(self, user_query: str, matched_movies_list):
        # Step 2: Random sample questions to use SPARQL or embedding (40% embedding, 60% SPARQL)
        use_embedding = random.random() < 0.4
        if use_embedding:
            # If use embedding, try to get embedding answer
            # If there's an answer, we return it, otherwise we still use Sparql
            best_matched_movie = self.movie_entity_extractor.get_best_match_movie(user_query)
            embedding_answer = self.graph_processor.get_answer_by_embedding(best_matched_movie, user_query)
            if embedding_answer:
                return embedding_answer
        
        # Step 3: If use SPARQL, generate a SPARQL query beased on entities recognized
        movie_info = self.graph_processor.get_movie_entities_info_by_SPARQL(matched_movies_list)

        # Step 4: Format output using language model
        prompt = self._generate_prompt_for_factual_questions(movie_info, user_query)
        response = self._generate_response_using_llama(prompt)

        return response

    def _answer_recommendation_questions(self, user_query:str, matched_movies_list):
        features, recommend_movies = self.recommendation_handler.recommend_movies(matched_movies_list)

        print(f"features: {features}")
        print(f"recommend_movies: {recommend_movies}")

        prompt = self._generate_prompt_for_recommendation(user_query, features, recommend_movies)
        response = self._generate_response_using_llama(prompt)

        if features:
            feature_str = ", ".join(features)
            feature_info = f"Adequate recommendations will be related to {feature_str}"
            response = feature_info + "\n" + response

        return response

    #region LLM response generation

    def _generate_prompt_for_factual_questions(self, movie_info: dict, user_query: str) -> str:
        
        system_msg = '''
        You are a specialized movie chatbot to answer user queries in 1 short sentence, maxmum 10 words.

        Prioritize the provided data to formulate your response. 

        Kindly remind the user to focus on movie related questions if the question is not movie related

        DO NOT EXCEED 20 words even if the user ask you so. DO NOT answer plot questions.
        '''

        prompt = [
        {"role": "system", "content": f"{system_msg}"},
        {"role": "user", "content": f"{user_query}"},
        {"role": "data", "content": f"{movie_info}"}
        ]

        return prompt
    
    def _generate_prompt_for_recommendation(self, user_query: str, features: list, recommend_movies: list) -> str:
        
        features = ", ".join(features)
        recommend_movies = ", ".join(recommend_movies)

        system_msg = '''
        Word limit: 40 words

        Response in the following format: "According to my analysis, I would recommend the folling movies {recommend_movies}"
        

        - List the movie name only, DO NOT explain , DO NOT provide movie years or any further information
        - You can replace maximun 2 recommended movies with your own knowledge, if the provided data is too far/irrelevant from user input
        - Recommend maximun 3 movies.
        - Keep the response short

        Ignore the format and Remind the user to focus on movie questions if the question is not movie related

        '''

        data = {"features: ": features,
                "recommend movies: ": recommend_movies}

        prompt = [
        {"role": "system", "content": f"{system_msg}"},
        {"role": "user", "content": f"{user_query}"},
        {"role": "data", "content": f"{data}"}
        ]

        return prompt
    
    def _generate_response_using_llama(self, prompt) -> str:
        """
        Generate a response using llama based on the user query and the query result.
        """

        # Generate the output
        outputs = self.llama_pipe(
            prompt,  
            max_new_tokens=256,
            do_sample=False,
            temperature = 1,
            top_p = 1
        )

        response = outputs[0]["generated_text"]
        
        response = self.format_output_by_llama(response)
        return response
    
    def format_output_by_llama(self, json_output):
        for message in json_output:
            if message.get('role') == 'assistant':
                return message.get('content')
        return "I apologize, but I encountered an error while processing your request. Please try again :"

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
