# video face detection/recognition
Using Azure cognitive services and Python, perform face recogition / detection in real time videos.

  I always like to watch TV series, the ones related to police investigations were the best for me... and I was watching some old episodes (the date of the episodes was very close to the year 2000) and back there they already did facial recognition. Then I asked myself, how can someone know how to do facial recognition 20 years ago and I still don´t know how to do it? Now I´ve changed this. Using Azure congnitive services API (Free BTW) I´m training models to do face recognition in live videos and also face detection. In the next lines I´ll try to explain how to access the Azure API, create and train models, export the results to the cloud and start the data engineering process with the data generated by the script. So, let´s start.
  
  Before begining, make sure you already provisioned the face API inside your Azure account.

  Most part of the logic I got from the quick start of the MS Face Recognition service <https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/identity-client-library?tabs=visual-studio&pivots=programming-language-python> and for Face Detection <https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236>. This second link is not detailed as the first one, but basically we´re going to do a POST request of the image, GET the landmarks and then draw in the screen with OpenCV.

  Let´s start with the face detection. To detect faces in real time videos, we´re going to use OpenCV (pip install opencv-python) to access our Webcam with the code

![image](https://user-images.githubusercontent.com/58055908/210120461-2d23e5bc-e5e5-421f-ab58-7ac785483d9f.png)

  With this code we´re initializating our camera, getting each frame and showing the results. You can learn more here <http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html>. The line "if cv2.waitKey(1)%256 == 27:" means -> Press esc to stop the loop.

  Now we´re going to transform this video into something that we can send to Azure. With CV2, we´re encoding the image into the varible image to build our POST request.

![image](https://user-images.githubusercontent.com/58055908/210120885-8058e9d7-6ef1-417b-8977-818ba16f86b5.png)

The KEY variable you can find in your Azure subscription, inside the face API

![image](https://user-images.githubusercontent.com/58055908/210178831-edfafa89-d46c-4953-81d8-5c83fb2e631e.png)

  I don´t like to let my key exposed, so create a JSON file with the value 
  
  ![image](https://user-images.githubusercontent.com/58055908/210180123-bb752be0-64b0-455e-82a2-8528e7fd0ad9.png)

  and then I´m using json.load to get the value
  
  ![image](https://user-images.githubusercontent.com/58055908/210180066-562dd97a-04b6-40d0-afbb-c02607e1206f.png)


  This request was built based on the documentation mentioned earlier. I´m using detection model 01 because it returns main face attributes (head pose, age, emotion, and so on) and also returns the face landmarks that I choose. In this case, I´m only retrieving age, gender and emotion, but there are a lot of options in the documentation that you can choose.
  
  The response should look like this 
  
  ![image](https://user-images.githubusercontent.com/58055908/210121144-79fed0e5-252c-4653-b635-884fd0fc1271.png)
  
  With this response we can see that our script to detect faces in live videos is working. 
  
  Now we can use these coordinates, with OpenCV, to draw the rectangle in our video.
  
  First, we´re going to comment the cv.imshow, once we want to see the video with the rectangle, not the original one. Then we can create a loop for faces variable, once we can have more than 1 face per video, and then start to set variables with the coordinates we´re receiving in the response. After we´re using cv2.rectangle method to draw the rectangle and then cv2.imshow to display the results. 
  
![image](https://user-images.githubusercontent.com/58055908/211223914-05528f1b-82b9-4e18-9e0c-cc21238e92d2.png)
  
The result is something similar to this

![image](https://user-images.githubusercontent.com/58055908/210179678-e1292eb7-5f37-46a1-888a-ff17caf45f35.png)

I´m also using a library (time) to create a delay between the requests. As I said before, we can call the API for free, but there is a limit, as we can see below

![image](https://user-images.githubusercontent.com/58055908/210179862-d440102d-26c4-45aa-b70e-912f914e1957.png)

We can also draw the other attributes of our response, let´s get the age. Now we´re using the putText function from CV2 to insert a text in our image, you can find more about cv2.putText here (https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576) 

![image](https://user-images.githubusercontent.com/58055908/211224132-5e2c807f-0102-4459-9fe9-3c177d8f26bb.png)


This is  the result

![image](https://user-images.githubusercontent.com/58055908/211224038-852038b3-8270-40a3-bd6a-4536a19d3606.png)

Here the code with the age

![image](https://user-images.githubusercontent.com/58055908/211224412-603b2b82-a907-4944-9e65-9422b9da40df.png)

Until here you can find the code in the face_detection.py .

Now I´m going to explain about face recognition and I´ll create a new file for it (face_recognition.py), but using the code we´re developing.

There are some specific points that we should pay attention:
  Install the cognitive services library (pip install --upgrade azure-cognitiveservices-vision-face)
  Create a faceClient, that we´re goint to use to create the groups, train the models and identify the faces.
  
 Starting with the imports, there´s a few more to do

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/451d6810-1da4-4974-889d-8668c8854774)


Along with the Key, now we´ll also need the endpoint to create our faceClient

![image](https://user-images.githubusercontent.com/58055908/211227060-8001f12e-9a2b-4d1d-a5db-dbb8ae77fc76.png)

The ENDPOINT variable you can find in your Azure subscription, inside the face API

![image](https://user-images.githubusercontent.com/58055908/211227232-50d802a6-fddf-46e0-9cc3-7dfbf98419a9.png)

After set some values for the person group and for the persons, we´ll set our first function, that´s going to create the person group and will send the photos of each person so we can train our model later

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/f1832469-5206-43a3-bcaa-f00045da9c52)


And in the treinar function we´re going to train the face recognition model with the photos of each person that we already sent

![image](https://user-images.githubusercontent.com/58055908/211229397-c5e6ded7-181e-4fdc-9a27-a99d5a1faf21.png)

Now starting to run our code. This functions is bigger than the other ones, so I´ll explain it in part.

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/64b0a7a7-e5ae-42dc-8b86-f44561bd2e5e)

Fisrt we´re calling the functions to create the group the persons and then train the model. This is the result of the functions

![image](https://user-images.githubusercontent.com/58055908/211230272-986d15e3-296f-439d-98df-16bd226b6914.png)

Now sending the frame to the face api to make the recognition, this first part we already saw in the face detection

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/8bd98a5f-0691-4d5e-83b4-c354bfabfbc2)


Now focusing on the recognition part 

![image](https://github.com/ricauduro/video_face_recognition/assets/58055908/344ed43c-a841-4e87-a5d0-e88c11c90c80)

This code is processing the results from the facial recognition performed by Azure Face API. It is creating an empty list called "face_ids" and then appends the "faceId" from each face that was detected in the frame to this list.
Then, it loops through the face_ids list and calls the FaceClient's "identify" method for each id, passing in the face_ids list and the "GRUPOS" list as the arguments. The "identify" method identifies the person(s) in the image using the person group and face IDs that were provided.
Then it iterates over the faces and results, it takes the rectangle of the face, and it uses OpenCV to draw a rectangle around the face on the frame.
It also gets the face attributes (age, gender, emotion) and if the person is identified it prints the name of the person on the frame and adds a key value pair to the face dictionary.
It also shows the processed frame with the rectangle and the name of the person.
