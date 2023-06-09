# Imports
import cv2
import requests
import time
import json

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import glob
import sys
from datetime import datetime
import os

# Variables
path = 'C:\\Users\\ricardo.cauduro\OneDrive - Kumulus\\Desktop\\Data\\NTB'
source_file = f'{path}\\FaceVideoDetection\\output\\mydata-'

credential = json.load(open(f'{path}\\key.json'))
KEY = credential['KEY']
ENDPOINT = credential['ENDPOINT']

face_api_url = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect"
headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
params = {'detectionModel': 'detection_01', 'returnFaceId': 'true', 'returnFaceRectangle': 'true', 'returnFaceAttributes': 'age, gender, emotion'}

GRUPOS = ['familia']
PESSOAS = ['ricardo', 'rita']
ID = []

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

def iniciar():
    cam = cv2.VideoCapture(0)
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    list(map(lambda x: criar_grupo(x), GRUPOS))
    list(map(lambda x: criar_pessoa(x,'familia'), PESSOAS))
    list(map(lambda x: treinar(x), GRUPOS))

    while True:
        data_insert = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        ret, frame = cam.read()
        image = cv2.imencode('.jpg', frame)[1].tobytes()

        response = requests.post(face_api_url, params=params, headers=headers, data=image)
        response.raise_for_status()
        faces = response.json()
        face_ids = [face['faceId'] for face in faces]

        global results
        for face in face_ids:
            results = face_client.face.identify(face_ids, 'familia')

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

        k = cv2.waitKey(1) & 0xFF # bitwise AND operation to get the last 8 bits
        if k == 27:
            print("Escape hit, closing...")
            break

def fim():
    cv2.destroyAllWindows()
    face_client.person_group.delete(person_group_id='familia')

# Start the code
iniciar()

# Stop and clean
fim()
