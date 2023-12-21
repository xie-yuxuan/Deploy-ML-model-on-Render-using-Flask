from flask import Flask, render_template, request # Flask class instance will be a WSGI (Web Server Gateway Interface) application

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# create an instance by providing __mame__ which is the name of the current module
app = Flask(__name__)
model = VGG16()

# register a route: URL pattern that an application can respond to
@app.route('/', methods=['GET']) # decorator for routing, get request to '/' which is homepage
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST']) # post request to homepage
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename # get images in image path
    imagefile.save(image_path)

    # load and preprocess the image to model requirements 
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # model makes prediction
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    # returns prediction here
    classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    # parse output into template
    return render_template('index_html', prediction=classification)

# run flask application, '__main__' to ensure server runs if script is executed directly, not imported  
if __name__ == '__main__':
    app.run(port=3000,debug=True)
