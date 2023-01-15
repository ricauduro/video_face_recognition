# Imports
import cv2
import requests
import time
import json

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import glob
import sys

cam = cv2.VideoCapture(0)

path = ''

credential = json.load(open(path))
KEY = credential['KEY']
ENDPOINT = credential['ENDPOINT']

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

GRUPOS = ['familia']
PESSOAS = ['ricardo', 'rita']
ID = []


# Functions
def criar_pessoa(pessoa):
    globals()[pessoa] = face_client.person_group_person.create(GRUPOS[0], pessoa)
    print('Person ID:', globals()[pessoa].person_id)
    ID.append(globals()[pessoa].person_id)

    listaFotos = [file for file in glob.glob('*.jpg') if file.startswith(pessoa)]
    time.sleep (1)
    for image in listaFotos:
        face_client.person_group_person.add_face_from_stream(
            GRUPOS[0], globals()[pessoa].person_id, open(image, 'r+b'))
        print(f'Incluida foto {image}')
        time.sleep (1)


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


face_client.person_group.create(person_group_id=GRUPOS[0], name=GRUPOS[0])
print(f'Criado o grupo {GRUPOS[0]}')

for pessoa in PESSOAS:
    criar_pessoa(pessoa)

treinar(GRUPOS[0])


while True:
    ret, frame = cam.read()

    if cv2.waitKey(1)%256 == 27:
        break

    image = cv2.imencode('.jpg', frame)[1].tobytes()
    
    face_api_url = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect"
    headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': KEY}
    params = {'detectionModel': 'detection_01', 'returnFaceId': 'true', 'returnFaceRectangle': 'true', 'returnFaceAttributes': 'age, gender, emotion'}

    response = requests.post(face_api_url, params=params, headers=headers, data=image)
    
    response.raise_for_status()
    faces = response.json()

    face_ids = []
    global results
    results = []
    
    for face in faces:
        face_ids.append(face['faceId'])
    
    for face in face_ids:
        results = face_client.face.identify(face_ids, GRUPOS[0])

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
    time.sleep(5)

cam.release()
cv2.destroyAllWindows()
