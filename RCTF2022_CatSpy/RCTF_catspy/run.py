from flask import Flask, render_template, request
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from flag import flag

def checkImg(Img):
    im = Image.open('static/start.png').convert('RGB')
    Img = Img.convert('RGB')
    if Img.size != (60, 60):
        return 0
    count = 0
    for i in range(60):
        for j in range(60):
            if im.getpixel((i,j)) != Img.getpixel((i,j)):
                count += 1
    if count == 1:
        return 1
    else:
        return 0

def divide(img):
  # Step 1: Initialize model with the best available weights
  weights = ResNet50_Weights.DEFAULT
  model = resnet50(weights=weights)
  model.eval()

  # Step 2: Initialize the inference transforms
  preprocess = weights.transforms()

  # Step 3: Apply inference preprocessing transforms
  batch = preprocess(img).unsqueeze(0)

  # Step 4: Use the model and print the predicted category
  prediction = model(batch).squeeze(0).softmax(0)
  class_id = prediction.argmax().item()
  score = prediction[class_id].item()
  category_name = weights.meta["categories"][class_id]
  return category_name,score


app = Flask(__name__)
@app.route('/', methods=['POST', 'GET'])
def welcome():
    return render_template("index.html")

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        im = Image.open(f)
        if checkImg(im) == 0:
            return render_template('upload.html', error="image format error! the image size must be 60 x 60 and you can only change one pixel!")
        category_name,score = divide(im)
        if category_name == 'tabby' or "cat" in category_name:
            return render_template('upload.html', res=category_name + "  " + str(score))
        else:
            return render_template('upload.html', flag=flag)
    return render_template('upload.html',error='please start attack!')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)