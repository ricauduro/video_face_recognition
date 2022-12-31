# video_face_recognition
Using Azure cognitive services and Python, do face recogition / detection in real time videos.

  I always like to watch TV series, the ones related to police investigations were the best for me... and I was watching some old episodes of a serie (the date of the episodes was very close to the year 2000) and back there they already did facial recognition. Then I asked myself, how can someone know how to do facial recognition 20 years ago and I still don´t know how to do it? Now I´ve changed this. Using Azure congnitive services API (Free BTW) I´m training models to do face recognition in live videos and also face detection. In the next lines I´ll try to explain how to access the Azure API, create and train models, export the results to the cloud and start the data engineering process with the data generated by the script. So, let´s start.
  
  Before begining, make sure you already provisioned the face API inside your Azure account.

  Most part of the logic I got from the quick start of the MS Face Recognition service <https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/identity-client-library?tabs=visual-studio&pivots=programming-language-python> and for Face Detection <https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236>. This second link is not detailed as the first one, but we´re going to do a POST request of the image, GET the landmarks and then draw in the screen with OpenCV.

  Let´s start with the face detection. To detect faces in real time videos, we´re going to use OpenCV (pip install opencv-python) to access our Webcam with the code

![image](https://user-images.githubusercontent.com/58055908/210120461-2d23e5bc-e5e5-421f-ab58-7ac785483d9f.png)

  With this code we´re initializating our camera, getting each frame and showing the results. You can learn more here <http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html>. The line "if cv2.waitKey(1)%256 == 27:" means -> Press esc to stop the loop.

  Now we´re going to transform this video into something that we can send to Azure. With CV2, we´re encoding the image into the varible image to build our POST request. The KEY variable you can find in your Azure subscription, inside the face API

![image](https://user-images.githubusercontent.com/58055908/210120885-8058e9d7-6ef1-417b-8977-818ba16f86b5.png)

  This request was built based on the documentation mentioned earlier. I´m using detection model 01 because it returns main face attributes (head pose, age, emotion, and so on) and also returns the face landmarks that I choose.
  The response should look like this 
  
  ![image](https://user-images.githubusercontent.com/58055908/210121144-79fed0e5-252c-4653-b635-884fd0fc1271.png)
  
  With this response we can see that our script to detect faces in live videos is working. Now we can use these coordinates, with OpenCV, to draw the rectangle in our video.
