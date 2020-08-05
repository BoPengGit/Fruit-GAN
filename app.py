import random
from flask import Flask

app = Flask(__name__)

@app.route("/fruit_gan/<fruit>", methods = ["GET"])
def fruit_gan(fruit):
    """Returns pre-generated fruit image based on fruit Name."""
    
    pass

