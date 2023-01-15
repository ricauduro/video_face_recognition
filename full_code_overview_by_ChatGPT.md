# Introduction

This script is a Python script that uses OpenCV, Azure SDK and json library to perform facial recognition. It captures video from a camera and then applies the Azure Face API to detect faces in the frames.

Setting up the environment
The script starts by importing the necessary libraries:

  **cv2** is used for capturing video from the camera and processing the frames.
  
  **requests** is used for making HTTP requests to the Azure Face API.
  
  **time** is used for adding delays between certain operations.
  
  **json** is used for loading the credentials from a json file.
  
  **FaceClient** and CognitiveServicesCredentials from the azure.cognitiveservices.vision.face package are used for connecting to the Azure Face API.
  
  **glob** is used for finding files in a directory.
  
  **sys** is used for exiting the script in case of errors.

A video capture object is created using cv2.VideoCapture(0), where 0 is the index of the camera that should be used. In this case, it is the default camera.

The script then loads the credentials from a json file located at path = 'c:\\Users\\ricardo.cauduro\\OneDrive - Kumulus\\Desktop\\Notebooks\\data\\key.json'

# Setting up the Face API
KEY and ENDPOINT variables are created from the credential file, and it creates a connection to Azure Face API with the provided credentials.

GRUPOS and PESSOAS are lists that contain the group and person names respectively.

ID is an empty list that will be used to store the person IDs.

# Functions
The script defines two functions:

criar_pessoa(pessoa): This function creates a new person in the person group and assigns the person ID to the ID list. It also loads all images with a specific name (e.g. "ricardo.jpg", "rita.jpg") and associates them with the corresponding person.

treinar(grupo): This function starts the training process for the person group. It checks the training status of the group and if it fails, it exits the script.

# Main Execution
The script first creates a person group with the name specified in the GRUPOS list.

It then iterates over the names in the PESSOAS list and calls the criar_pessoa() function for each name.

It then calls the treinar() function with the group name as the argument.

The script then enters a while loop that captures video frames and applies the Azure Face API to detect faces in the frames.

It first checks if the user pressed the 'ESC' key and if so, it breaks the loop.

It then gets the frame from the video capture object and encodes it as a jpeg.

It makes a post request to the Azure Face API with the image, headers and parameters required.

It raises an exception if the status code of the response is not 200

It then processes the results returned by the API, it creates an empty list called "face_ids" and then appends the "faceId" from each face that was detected in the frame to this list.

Then, it loops through the face_ids list and calls the FaceClient's "identify"
