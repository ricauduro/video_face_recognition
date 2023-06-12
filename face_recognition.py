# Imports
import cv2
import requests
import time
import json
import glob
import sys
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

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
