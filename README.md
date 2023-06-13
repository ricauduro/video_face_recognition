# video face detection/recognition
Using Azure cognitive services and Python, perform face recogition / detection in real time videos.

  I always like to watch TV series, the ones related to police investigations were the best for me... and I was watching some old episodes (the date of the episodes was very close to the year 2000) and back there they already did facial recognition. Then I asked myself, how can someone know how to do facial recognition 20 years ago and I still don´t know how to do it? Now I´ve changed this. Using Azure congnitive services API (Free BTW) I´m training models to do face recognition in live videos and also face detection. In the next lines I´ll try to explain how to access the Azure API, create and train models, export the results to the cloud and start the data engineering process with the data generated by the script. So, let´s start.
  
  Before begining, make sure you already provisioned the face API inside your Azure account.

  Most part of the logic I got from the quick start of the MS Face Recognition service <https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/identity-client-library?tabs=visual-studio&pivots=programming-language-python> and for Face Detection <https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236>. This second link is not detailed as the first one, but basically we´re going to do a POST request of the image, GET the landmarks and then draw in the screen with OpenCV.

  Let´s start with the face detection. To detect faces in real time videos, we´re going to use OpenCV (pip install opencv-python) to access our Webcam with the code

```Python
import cv2
import requests
import time
import json

cam = cv2.VideoCapture(0)

path = "Path where you saved your json file with your key"

credential = json.load(open(path))
KEY = credential['KEY']

while True:
    ret, frame = cam.read()
    # cv2.imshow('frame', frame)

    if cv2.waitKey(1)%256 == 27:
        break
```

  With this code we´re initializating our camera, getting each frame and showing the results. You can learn more here <http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html>. The line "if cv2.waitKey(1)%256 == 27:" means -> Press esc to stop the loop.

  Now we´re going to transform this video into something that we can send to Azure. With CV2, we´re encoding the image into the varible image to build our POST request.

```Python
import cv2
import requests
import time
import json

cam = cv2.VideoCapture(0)

path = "Path where you saved your json file with your key"

credential = json.load(open(path))
KEY = credential['KEY']

while True:
    ret, frame = cam.read()
    # cv2.imshow('frame', frame)

    if cv2.waitKey(1)%256 == 27:
        break

    image = cv2.imencode('.jpg', frame)[1].tobytes()
    
    face_api_url = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect"
    headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
    params = {'detectionModel': 'detection_01', 'returnFaceId': 'true', 'returnFaceRectangle': 'true', 'returnFaceAttributes': 'age, gender, emotion'}

    response = requests.post(face_api_url, params=params, headers=headers, data=image)
    
    response.raise_for_status()
    faces = response.json()
    print(faces)

cam.release()
cv2.destroyAllWindows()
```

The KEY variable you can find in your Azure subscription, inside the face API

![image](https://user-images.githubusercontent.com/58055908/210178831-edfafa89-d46c-4953-81d8-5c83fb2e631e.png)

  I don´t like to let my key exposed, so create a JSON file with the value 
  
  ![image](https://user-images.githubusercontent.com/58055908/210180123-bb752be0-64b0-455e-82a2-8528e7fd0ad9.png)

  and then I´m using json.load to get the value
  
```Python
path = "Path where you saved your json file with your key"

credential = json.load(open(path))
KEY = credential['KEY']
```


  This request was built based on the documentation mentioned earlier. I´m using detection model 01 because it returns main face attributes (head pose, age, emotion, and so on) and also returns the face landmarks that I choose. In this case, I´m only retrieving age, gender and emotion, but there are a lot of options in the documentation that you can choose.
  
  The response should look like this 
  
  ![image](https://user-images.githubusercontent.com/58055908/210121144-79fed0e5-252c-4653-b635-884fd0fc1271.png)
  
  With this response we can see that our script to detect faces in live videos is working. 
  
  Now we can use these coordinates, with OpenCV, to draw the rectangle in our video.
  
  First, we´re going to comment the cv.imshow, once we want to see the video with the rectangle, not the original one. Then we can create a loop for faces variable, once we can have more than 1 face per video, and then start to set variables with the coordinates we´re receiving in the response. After we´re using cv2.rectangle method to draw the rectangle and then cv2.imshow to display the results. 
  
```Python
while True:
    ret, frame = cam.read()
    # cv2.imshow('frame', frame)

    if cv2.waitKey(1)%256 == 27:
        break

    image = cv2.imencode('.jpg', frame)[1].tobytes()
    
    face_api_url = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect"
    headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
    params = {'detectionModel': 'detection_01', 'returnFaceId': 'true', 'returnFaceRectangle': 'true', 'returnFaceAttributes': 'age, gender, emotion'}

    response = requests.post(face_api_url, params=params, headers=headers, data=image)
    
    response.raise_for_status()
    faces = response.json()
    print(faces)

    for face in faces:
        rect = face['faceRectangle']
        left = rect['left']
        top = rect['top']
        right = int(rect['width']) + int(rect['left'])
        bottom = int(rect['height']) + int(rect['top'])

        draw = cv2.rectangle(frame,(left, top), (right, bottom),(0, 255, 0), 3)
       
    cv2.imshow('face_rect', draw)
    time.sleep(3)

cam.release()
cv2.destroyAllWindows()
```
  
The result is something similar to this

![image](https://user-images.githubusercontent.com/58055908/210179678-e1292eb7-5f37-46a1-888a-ff17caf45f35.png)

I´m also using a library (time) to create a delay between the requests. As I said before, we can call the API for free, but there is a limit, as we can see below

![image](https://user-images.githubusercontent.com/58055908/210179862-d440102d-26c4-45aa-b70e-912f914e1957.png)

We can also draw the other attributes of our response, let´s get the age. Now we´re using the putText function from CV2 to insert a text in our image, you can find more about cv2.putText here (https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576) 

```Python
att = face['faceAttributes']
age = att['age']

draw = cv2.putText(draw, 'Age: ' + str(age), (left, bottom + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
 ```


This is  the result

![image](https://user-images.githubusercontent.com/58055908/211224038-852038b3-8270-40a3-bd6a-4536a19d3606.png)

Here the code with the age

```Python
while True:
    ret, frame = cam.read()
    # cv2.imshow('frame', frame)

    if cv2.waitKey(1)%256 == 27:
        break

    image = cv2.imencode('.jpg', frame)[1].tobytes()
    
    face_api_url = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect"
    headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
    params = {'detectionModel': 'detection_01', 'returnFaceId': 'true', 'returnFaceRectangle': 'true', 'returnFaceAttributes': 'age, gender, emotion'}

    response = requests.post(face_api_url, params=params, headers=headers, data=image)
    
    response.raise_for_status()
    faces = response.json()
    print(faces)

    for face in faces:
        rect = face['faceRectangle']
        left = rect['left']
        top = rect['top']
        right = int(rect['width']) + int(rect['left'])
        bottom = int(rect['height']) + int(rect['top'])

        draw = cv2.rectangle(frame,(left, top), (right, bottom),(0, 255, 0), 3)

        att = face['faceAttributes']
        age = att['age']

        draw = cv2.putText(draw, 'Age: ' + str(age), (left, bottom + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
    cv2.imshow('face_rect', draw)
    time.sleep(3)

cam.release()
cv2.destroyAllWindows()
```

Until here you can find the code in the face_detection.py .

Now I´m going to explain about face recognition and I´ll create a new file for it (face_recognition.py).

There are some specific points that we should pay attention:
  Install the cognitive services library (pip install --upgrade azure-cognitiveservices-vision-face)
  Create a faceClient, that we´ll need to create the groups, train the models and identify the faces.
  
 Starting with the imports, there´s a few more to do

```Python
# Imports
import cv2
import requests
import time
import json
import glob
import sys
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
```


Along with the Key, now we´ll need to define some lists to store our groups, persons, id´s and also need the endpoint to create our faceClient, 

```Python
# Variables
path = 'C:\\Users\\ricardo.cauduro\OneDrive - Kumulus\\Desktop\\Data\\NTB'

credential = json.load(open(f'{path}\\key.json'))
KEY = credential['KEY']
ENDPOINT = credential['ENDPOINT']

face_api_url = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect"
headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
params = {'detectionModel': 'detection_01', 'returnFaceId': 'true', 'returnFaceRectangle': 'true', 'returnFaceAttributes': 'age, gender, emotion'}

GRUPOS = []
PESSOAS = []
ID = []

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
```

The ENDPOINT variable you can find in your Azure subscription, inside the face API

![image](https://user-images.githubusercontent.com/58055908/211227232-50d802a6-fddf-46e0-9cc3-7dfbf98419a9.png)

Now with these functions we´ll create the person groups, assign each person to a person group and then send the photos of each person so we can train our model later

```Python
# Functions
def criar_grupo(grupo):
    face_client.person_group.create(person_group_id=grupo, name=grupo)
    print(f'Criado o grupo {grupo}')

def criar_pessoa(pessoa, grupo):
    globals()[pessoa] = face_client.person_group_person.create(grupo, pessoa)
    print('Person ID:', globals()[pessoa].person_id)
    ID.append(globals()[pessoa].person_id)

    listaFotos = [file for file in glob.glob('*.jpg') if file.startswith(pessoa)]
    time.sleep(1)
    for image in listaFotos:
        face_client.person_group_person.add_face_from_stream(
            GRUPOS[0], globals()[pessoa].person_id, open(image, 'r+b'))
        print(f'Incluida foto {image}')
        time.sleep(1)
```
And in the treinar function we´re going to train the face recognition model with the photos of each person that we already sent

```Python
def treinar(grupo):
    print(f'Iniciando treino de {grupo}')
    face_client.person_group.train(grupo)
    while (True):
        training_status = face_client.person_group.get_training_status(grupo)
        print("Training status de {}: {}.".format(grupo, training_status.status))
        if (training_status.status == 'succeeded'):
            break
        elif (training_status.status == 'failed'):
            face_client.person_group.delete(person_group_id=grupo)
            sys.exit('Training the person group has failed.')
        time.sleep(5)
```

Now starting to run our code. First we´ll set the name of the person group and os each person in this group using the input, then we´ll create the groups, the persons and train the model with the pictures saved on our local folder, and then we´ll start the camera.

def iniciar():
    GRUPOS.append(input('Defina o nome do grupo -> ').lower())
    list(map(lambda x: criar_grupo(x), GRUPOS))
    
    lista_pessoas = []
    nome_pessoa = None
    while nome_pessoa != 'fim':
        nome_pessoa = input(f"Digite o nome da pessoa para associar ao grupo '{GRUPOS[0]}' ou digite 'fim' para terminar. -> ").lower()
        if nome_pessoa != 'fim':
            PESSOAS.append(nome_pessoa)
            lista_pessoas.append(nome_pessoa)
    
    if len(lista_pessoas) == 1:
        print('{0} foi adicionado ao grupo {1}'.format(PESSOAS[0], GRUPOS[0]))
    else:
        ultimo_nome = lista_pessoas.pop()
        nomes = ', '.join(lista_pessoas)
        print('{0} e {1} foram adicionados ao grupo {2}'.format(nomes, ultimo_nome, GRUPOS[0]))

    list(map(lambda x: criar_pessoa(x,'familia'), PESSOAS))
    list(map(lambda x: treinar(x), GRUPOS))

    cam = cv2.VideoCapture(0)

This is the result of the functions

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/b1f13ffa-28bb-4292-9a8a-870f362be91d)

Now sending the frame to the face api to make the recognition, this first part we already saw in the face detection

```Python
    while True:
        ret, frame = cam.read()
        image = cv2.imencode('.jpg', frame)[1].tobytes()

        response = requests.post(face_api_url, params=params, headers=headers, data=image)
        response.raise_for_status()
        faces = response.json()
        face_ids = [face['faceId'] for face in faces]

        global results
        for face in face_ids:
            results = face_client.face.identify(face_ids, 'familia')
```


Now focusing on the recognition part 

```Python
        # Obtendo landmarks
        for face, person in zip(faces, results):
            rect = face['faceRectangle']
            left = rect['left']
            top = rect['top']
            right = int(rect['width']) + int(rect['left'])
            bottom = int(rect['height']) + int(rect['top'])

            draw = cv2.rectangle(frame,(left, top), (right, bottom),(0, 255, 0), 3)
            att = face['faceAttributes']
            age = att['age']

            # Person recognition
            for id, nome in zip(ID, PESSOAS):
                if len(person.candidates) > 0 and str(person.candidates[0].person_id) == str(id):
                    print('Person for face ID {} is identified in {}.{}'.format(person.face_id, 'Frame',person.candidates[0].person_id))
                    draw = cv2.putText(frame, 'Nome: ' + nome, (left, bottom + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    faces[0]['nome'] = str(nome)
                else:
                    draw = cv2.putText(frame, 'Nome: ' + '', (left, bottom + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0,  255), 1, cv2.LINE_AA)
            cv2.imshow('face_rect', draw)
```

This code is processing the results from the facial recognition performed by Azure Face API. It is creating an empty list called "face_ids" and then appends the "faceId" from each face that was detected in the frame to this list.
Then, it loops through the face_ids list and calls the FaceClient's "identify" method for each id, passing in the face_ids list and the "GRUPOS" list as the arguments. The "identify" method identifies the person(s) in the image using the person group and face IDs that were provided.
Then it iterates over the faces and results, it takes the rectangle of the face, and it uses OpenCV to draw a rectangle around the face on the frame.
It also gets the face attributes (age, gender, emotion) and if the person is identified it prints the name of the person on the frame and adds a key value pair to the face dictionary.
It also shows the processed frame with the rectangle and the name of the person, like this

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/49fff001-af0f-4905-8bfb-97e9ba06fbd1)


Once I´m explain how to perform face recognition on live videos, I´ll record a video with the full code so you can see it in live action. Here´s the link https://youtu.be/6Sx00lH1mTE



