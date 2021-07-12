from flask import Flask, render_template, request
# Flask-It is our framework which we are going to use to run/serve our applicatio #request-for accessing file which was uploaded by the user on our application.
import operator
import cv2 # opencv library
from tensorflow.keras.models import load_model#to load our trained model
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__,template_folder="templates")# initializing a flask app 
model=load_model('gesture.h5')
print("Loaded model from disk")

@app.route('/')
def home():
    return render_template( 'home.html') #rendering the home page

@app.route('/intro') # routes to the intro page 
def intro():
    return render_template('intro.html')#rendering the intro page

@app.route('/image1', methods=['GET', 'POST']) # routes to the index html 
def image1():
    return render_template("index6.html")
@app.route('/predict',methods=['GET', 'POST'])# route to show the predictions in a web UI 
def predict():
    if request.method == 'POST':
        print("inside image")
        f = request.files['image']
        print(f)
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename (f.filename))
        f.save(file_path)
        print(file_path)
        cap = cv2.VideoCapture (0)

        while True:
            _,frame = cap.read() #capturing the video frame values
            # Simulating mirror image 
            frame = cv2.flip(frame, 1)
            # Got this from collect-data.py
            # Coordinates of the ROI
           
            
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2= int(0.5*frame.shape[1])
            #the increment/decrement by 1 is to composefor the translate for the bounding box
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0),1)
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (64, 64))
            roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _,test_image= cv2.threshold (roi, 120, 255, cv2.THRESH_BINARY)
            # cv2.imshow("test", test_image)

            roi2 = frame[y1:y2, x1:x2]
            roi2 = cv2.resize(roi2, (500, 500))
            roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            _, test_image2 = cv2.threshold(roi2, 120, 255, cv2.THRESH_BINARY)
            cv2.imshow("test", test_image2)

            test_image = cv2.bitwise_not(test_image)
            result =  model.predict(test_image.reshape (1, 64, 64, 1))
            print(result)
        
            prediction ={'ZERO': result[0][0],
                         'ONE': result[0][1],
                         'TWO': result[0][2],
                         'THREE': result[0][3],
                         'FOUR': result[0][4],
                         'FIVE': result[0][5]}
                
            #Sorting based on top prediction
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            # Displaying the predictions
            cv2.putText(frame, prediction [0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,64,64), 1)
            #loading an image
            image1=cv2.imread(file_path)
            if prediction[0][0]== 'ONE':
                resized =  cv2.resize(image1, (200, 200)) 
                cv2.imshow("output", resized)
                time.sleep(0.5)

                key=cv2.waitKey(10)

                if (key & 0xFF) == ord("1"):
                    cv2.destroyWindow("output")
            elif prediction[0][0]== 'ZERO':
                cv2.rectangle(image1, (480, 170), (650, 420), (0, 0, 255), 2) 
                cv2.imshow("output", image1)

                key=cv2.waitKey(10)
                if (key & 0xFF) == ord("0"):
                    cv2.destroyWindow("output")
            elif prediction[0][0]=='TWO':
                (h, w, d)= image1.shape
                center =  (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -45, 1.0) 
                rotated =  cv2.warpAffine (image1, M, (w, h)) 
                cv2.imshow("output", rotated)


                key=cv2.waitKey(10)
                if (key & 0xFF) == ord("2"):
                    cv2.destroyWindow("output")
            elif prediction[0][0]== 'THREE':
                blurred= cv2.GaussianBlur (image1, (11, 11), 0)
                cv2.imshow("output", blurred)
                time.sleep(0.5)

                key=cv2.waitKey(10)
                if (key & 0xFF) == ord("3"):
                    cv2.destroyWindow("output")
            else:
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
if __name__ == "__main__":
    app.run(debug=True)