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
        # Build a mapping from movie id to list of image entries
        self.movie_to_images = {}  # Key: movie_id, Value: list of image entries
        for entry in self.data:
            movies = entry.get('movie', [])
            for movie_id in movies:
                if movie_id not in self.movie_to_images:
                    self.movie_to_images[movie_id] = []
                self.movie_to_images[movie_id].append(entry)
        

    def show_image(self, user_query:str, movie_name: str)-> str:
        
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
    
    def _get_image_id_from_imdb_id_dummy(self, imdb_id):
        # TODO implement this function, Note that the json data is large, the search should be effcient. 
        # Note that there might be multiple match for a movie imdb_id and img, and one img might has multiple movies as well
        # it is a many to many relation ship. 
        # For a given movie imdb, We will first check if 'type' is 'poster' and try to use it, if not usable (not exist)
        # If no 'poster' type images, fall back to 'publicity'
        pass
    
    def _get_imdb_id(self, movie_name: str):
        
        imdb_id = self.graph_processor._get_imdb_id_from_graph(movie_name)
        if not imdb_id:
            print(f"No IMDb ID found for the movie: {movie_name}")
            return 0
        return imdb_id