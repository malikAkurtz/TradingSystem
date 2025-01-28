import os
from dotenv import load_dotenv

load_dotenv()
# Getting all of our credentials from our local .env file
API_KEY = os.getenv('api_key')
API_SECRET = os.getenv('api_secret')
BASE_URL = os.getenv('base_url')
