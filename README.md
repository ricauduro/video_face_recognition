# video face detection/recognition
Using Azure cognitive services and Python, perform face recogition / detection in real time videos.

  I always like to watch TV series, the ones related to police investigations were the best for me... and I was watching some old episodes (the date of the episodes was very close to the year 2000) and back there they already did face recognition. Then I asked myself, how can someone know how to do face recognition 20 years ago and I still don´t know how to do it? Now I´ve changed this. Using Azure congnitive services API (Free BTW) I´m training models to perform face detection / recognition in live videos. In the next lines I´ll try to explain how to access the Azure API, create and train models. 
  
  So, let´s start.
  
## video face recognition
There are some specific points that we should pay attention:
  Install the cognitive services library (pip install --upgrade azure-cognitiveservices-vision-face)
  Create a faceClient, that we´ll need to create the groups, train the models and identify the faces.
  
 Starting with the imports

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


Along with the Key, we´ll also need to define some lists to store our groups, persons, id´s and also need the endpoint to create our faceClient, 

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

```Python
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
```

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
        for n, (face, person, id, nome) in enumerate(zip(faces, results, ID, PESSOAS)):
            rect = face['faceRectangle']
            left, top = rect['left'], rect['top']
            right = int(rect['width'] + left)
            bottom = int(rect['height'] + top)

            draw_rect = cv2.rectangle(frame,(left, top), (right, bottom),(0, 255, 0), 3)
            
            att = face['faceAttributes']
            age = att['age']

            if len(person.candidates) > 0 and str(person.candidates[0].person_id) == str(id):
                print('Person for face ID {} is identified in {}.{}'.format(person.face_id, 'Frame',person.candidates[0].person_id))
                draw_text = cv2.putText(frame, f'Nome:  {nome}', (left, bottom + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1,cv2.LINE_AA)
                faces[n]['nome'] = str(nome)
                
            else:
                draw_text = cv2.putText(frame,'Nome: Desconhecido', (left,bottom+50),cv2.FONT_HERSHEY_TRIPLEX , 0.5,(0 ,0 ,255 ),1,cv2.LINE_AA)
                faces[n]['nome'] ='Desconhecido'
        
        cv2.imshow('face_rect', draw_rect)
```

This code is processing the results from the facial recognition performed by Azure Face API. It is creating an empty list called "face_ids" and then appends the "faceId" from each face that was detected in the frame to this list.
Then, it loops through the face_ids list and calls the FaceClient's "identify" method for each id, passing in the face_ids list and the "GRUPOS" list as the arguments. The "identify" method identifies the person(s) in the image using the person group and face IDs that were provided.
Then it iterates over the faces and results, it takes the rectangle of the face, and it uses OpenCV to draw a rectangle around the face on the frame.
It also gets the face attributes (age, gender, emotion) and if the person is identified it prints the name of the person on the frame and adds a key value pair to the face dictionary.
It also shows the processed frame with the rectangle and the name of the person, like this

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/49fff001-af0f-4905-8bfb-97e9ba06fbd1)


Once I´m explain how to perform face recognition on live videos, I´ve recorded a video with the full code so you can see it in live action. Here´s the link https://youtu.be/6Sx00lH1mTE

## move data to blob storage
Now I´m going to explain how we can move the data we´re creating to a blob storage. I´ll create a new file for it (face_recognition_with_blob.py).

We´ll need 2 aditional imports

```Python
from azure.storage.blob import BlobServiceClient 
from datetime import datetime
```
We´ll also need to add some values to our key.json to use as credentials to connect to our blob storage.

```Python
storage_account_key = credential['storage_account_key']
storage_account_name = credential['storage_account_name']
connection_string = credential['connection_string']
container_name = credential['container_name']
```
And we´ll need another function to create the blob

```Python
def uploadToBlobStorage(file_path,file_name)
   blob_service_client = BlobServiceClient.from_connection_string(connection_string)
   blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
   with open(file_path,'rb') as data:
      blob_client.upload_blob(data)
      print('Uploaded {}.'.format(file_name))
```

Inside our code´s loop we´ll need to create two variables that we´ll use to create a folder for each day in the blob storage and another one to create the file name

```Python
    while True:
        folder_date = datetime.now().date().strftime('%Y%m%d')
        filename_date = datetime.now().strftime('%Y%m%d_%H%M%S')
```

Then for each face we´ll append a timestamp value, the bottomSize a location, we´ll save the file locally and then call the function to send the file to our blob

```Python
  # Gerando o arquivo com as informações captadas pelo video
  faces = [{**face, 'timeStamp': str(datetime.now()), 'bottomSize': str(bottom), 'location': 'Casa'} for face in faces]
  json_string = json.dumps(faces, separators=(',', ':'))

  with open('output\mydata-{}.json'.format(filename_date), 'w') as f:
            json.dump(json.JSONDecoder().decode(json_string), f)
            
  # Calling a function to perform upload
  uploadToBlobStorage('output\mydata-{}.json'.format(filename_date),'{}/mydata-{}.json'.format(folder_date,filename_date))
```
This is how the data should be in our storage

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/b84120f9-c0e0-4894-b0fa-7eb3fbd38ab3)

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/3eea1219-c3c1-4d66-a8ad-dd09db8376fd)

