# Anime_Classifer

This is a web app deployed on heroku using flask

**URL**: https://ani-car-class.herokuapp.com/

This web app classifies anime girls and cartoon girls and to an extent is able to classify anime and cartoons as well 

Dataset is self built using bing image api 

**Anime girl images** - 775

**Cartoon girl images** - 738

**Training set**:

Accuracy:  81.8 %
Loss: 0.35

**Testing set** (Used the testing set as the validation set for measuring overfitting):

Accuracy: 83.8 %
Loss: 0.34


------------------------------Future Changes-----------------------------------

Will use the initial layers of the pre trained network vgg16 as features for my network

Will remove the cache issue with the web app

Recreate an anime image using the art style learnt from the cartoon image and vice versa 

Can use a live camera for scanning images instead of uploading as a form

Upload the model onto a raspberry pi and use a webcam module for scanning images and display the classifier output on an LCD
