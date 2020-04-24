from flask import Flask

app = Flask(__name__)

from icapp import routes
