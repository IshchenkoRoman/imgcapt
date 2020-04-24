import sys
import os
from io import BytesIO
import base64

from PIL import Image

# I do not understand how properly import module
sys.path.append(os.path.join(os.getcwd(), "icapp", "NN"))

from flask import render_template, request
from icapp import app

from NN import get_image_captioning


IMAGE_SIZE = (640, 480)
SIZE_LIMIT = 6291456


app.config["IMAGE_UPLOADS"] = os.path.join(os.getcwd(), "icapp/img")
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024


print(os.getcwd())
BASE_IMAGE = Image.open(os.path.join(os.getcwd(), "img/intro.png")).resize(IMAGE_SIZE)
BASE_IMAGE = BASE_IMAGE.convert("RGB")
BASE_CAPTION = "Hello!" + "\n" + get_image_captioning(BASE_IMAGE)


def allowed_image(filename):

    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    return ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    """
    Main function that control behaviour for uploadimg image
    If file correct (size, name, type), then it passed through NN
    and predict caption for loaded picture.
    In other way used default picture

    One picture- one description.
    Image didn't saved in local storage

    Limit of size: 6mb (SIZE_LIMIT)
    """

    # Initialise base placeholders for description of image and image
    caption = ""
    image = BASE_IMAGE

    # Base on answer below
    # https://stackoverflow.com/questions/14348442/how-do-i-display-a-pil-image-object-in-a-template

    output = BytesIO()

    if request.method == "POST":

        if request.files:

            if request.content_length > SIZE_LIMIT:  # 6 Mb 6291456

                print("Filesize exceeded maximum limit")
                caption = f"Oh you! Too large file, limit: {SIZE_LIMIT} "\
                          f"You try load: {request.content_length}"
                # return redirect(request.url)

            else:

                image = request.files["image"]
                if image.filename == "":
                    print("No filename")
                    caption = "Oh you! No file name"
                    # return redirect(request.url)

                if allowed_image(image.filename):

                    # filename = image.filename
                    # image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                    capt = get_image_captioning(image)

                    caption = capt
                    image = Image.open(image)

                else:
                    image = BASE_IMAGE
                    print("That file extension is not allowed")
                    caption = "Oh you! Extension is not allowed"

    image = image.resize(IMAGE_SIZE)
    image.save(output, format="PNG")
    output.seek(0)

    image = "data:image/png;base64," + base64.b64encode(output.read()).decode()

    return render_template("index.html", caption=caption, figure=image)


@app.route("/")
@app.route("/index")
@app.route("/index.html")
def index():

    """
    Function for handling index.html loading

    It's just render index.html with base image and base caption of this image
    After that cache image and return it for displaying in index.html
    """

    image = BASE_IMAGE
    caption = BASE_CAPTION

    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)

    data = "data:image/png;base64," + base64.b64encode(output.read()).decode()

    return render_template("index.html", caption=caption, figure=data)


@app.route("/bio")
@app.route("/bio.html")
def bio():
    return render_template("bio.html")
