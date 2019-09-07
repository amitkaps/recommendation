import requests

tmdb_API_KEY = "4f23dc5304423b2d509195a634e7a702"

def movie(movie_id):
    """
    Gets the movie features from the tmdb api based on tmdb or imdb id
    
    tmdb_id: integer, tmdb id for the movie
    
    """
    url = "https://api.themoviedb.org/3/movie/" + str(movie_id)
    payload = {'api_key': tmdb_API_KEY, "language": "en_US"}
    response = requests.request("GET", url, params=payload)
    if response.status_code == 200:
        return response.json()
    else: 
        return False
    
    
    
def search(title, year=None):
    """
    Searches the tmdb api for the movie based on title
    
    title: string, title for the movie
    year: int, default=None, primary release year for the movie 
    
    """
    url = "https://api.themoviedb.org/3/search/movie/"
    payload = {'api_key': tmdb_API_KEY, 
               "language": "en_US", 
               "query": title, 
               "include_adult": False,
               "page": 1, 
               "primary_release_year": year}
    response = requests.request("GET", url, params=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return False

    