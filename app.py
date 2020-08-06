import random
import flask
from flask import request, jsonify

app = flask.Flask(__name__)

response = [{'class': 0, 'image': img}]

@app.route("/fruit_gan/<class>", methods = ["GET"])
def fruit_gan(fruit):
    """Returns pre-generated fruit image based on fruit Name."""    
        
    return jsonify(response)


app.run()