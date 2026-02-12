import pickle
from Agent.graph_processor import GraphProcessor

class MultimediaHandler:

    IMAGE_FILE_PATH = './../Dataset/images.pkl'

    def __init__(self, graph_processor: GraphProcessor):
        self.graph_processor = graph_processor
        
        # Load the data from the pickle file
        print("loading image net images.pkl")
        with open(self.IMAGE_FILE_PATH, 'rb') as f:
            self.data = pickle.load(f)
        print("images.pkl loaded")
        # Build a mapping from movie ID to list of image entries
        self.movie_to_images = {}  # Key: movie_id, Value: list of image entries
        # Build a mapping from person ID to list of image entries
        self.person_to_images = {}  # Key: person_id, Value: list of image entries
        
        print("Processing images data")
        for entry in self.data:
            # Map movie IDs to images
            movies = entry.get('movie', [])
            for movie_id in movies:
                if movie_id not in self.movie_to_images:
                    self.movie_to_images[movie_id] = []
                self.movie_to_images[movie_id].append(entry)
            
            # Map person IDs (cast) to images
            cast = entry.get('cast', [])
            for person_id in cast:
                if person_id not in self.person_to_images:
                    self.person_to_images[person_id] = []
                self.person_to_images[person_id].append(entry)
        print("Processed images data")

    def show_image_for_movie(self, user_query:str, movie_name: str)-> str:
        
        # Step 1: get imdb id from KG
        imdb_id = self._get_imdb_id(movie_name)

        if not imdb_id:
            return ""

        # Step 2: find image id using imdb id from image.json

        image_id = self._get_image_id_from_imdb_id(imdb_id)
        
        if not image_id:
            return ""
        
        image_id = image_id.split('.')[0]
        
        return image_id
    
    def show_image_for_person(self, user_query:str, person_name: str)-> str:
        
        # Step 1: get imdb id from KG

        imdb_id = self._get_imdb_id(person_name)
        if not imdb_id:
            return ""

        # Step 2: find image id using imdb id from image.json

        image_id = self._get_person_image_id_from_imdb_id(imdb_id)
        
        if not image_id:
            return ""
        
        image_id = image_id.split('.')[0]
        
        return image_id
    
    def _get_person_image_id_from_imdb_id(self, imdb_id):
        # Given a person imdb_id, find the img value according to the rules
        images = self.person_to_images.get(imdb_id, [])
        # First, check for images of type 'event'
        event_images = [img for img in images if img.get('type') == 'event']
        if event_images:
            # Return the 'img' value from the first event image
            return event_images[0]['img']
        else:
            # Check for images of type 'publicity'
            publicity_images = [img for img in images if img.get('type') == 'publicity']
            if publicity_images:
                # Return the 'img' value from the first publicity image
                return publicity_images[0]['img']
            else:
                # Return any available image if neither 'event' nor 'publicity' exist
                if images:
                    return images[0]['img']
                else:
                    # No image found
                    return None


    def _get_image_id_from_imdb_id(self, imdb_id):
        # Given an imdb_id, find the img value according to the rules
        images = self.movie_to_images.get(imdb_id, [])
        # First, check for images of type 'poster'
        poster_images = [img for img in images if img.get('type') == 'poster']
        if poster_images:
            # Return the 'img' value from the first poster image
            return poster_images[0]['img']
        else:
            # Check for images of type 'publicity'
            publicity_images = [img for img in images if img.get('type') == 'publicity']
            if publicity_images:
                # Return the 'img' value from the first publicity image
                return publicity_images[0]['img']
            else:
                # Return any available image if neither 'poster' nor 'publicity' exist
                if images:
                    return images[0]['img']
                else:
                    # No image found
                    return None
    
    def _get_imdb_id(self, lable: str):
        
        imdb_id = self.graph_processor._get_imdb_id_from_graph(lable)
        if not imdb_id:
            print(f"No IMDb ID found for the movie: {lable}")
            return 0
        return imdb_id