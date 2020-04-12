# age-emotion-gender-and-ethnicity-guesser-using-CNN-and-ML
An introductory project to machine learning and convolutional neural networks (CNNs). 
The project works with a webcam video feed to guess a person's age, emotion, gender, and ethnicity

This is my first machine learning-focused project so disclaimer: it’s going to be pretty terrible code, because 
I really only attempted to make it work rather than focusing on how to keep it organized and readable. 
There’s also not many comments sorry! But https://bluestampengineering.com/student-projects/aaron-s/ has 
several paragraphs where I explain some of the steps in the code and why I did each step.

There’s also a lot of junk files in the GitHub repository because I was pretty terrible with using Git.

I initially started off trying to create a smile detector, which later on I wanted to be more ambitious and created the emotion guesser. 
Here is the Jupyter notebook where I trained the model, and applied it
https://github.com/ashi72/age-emotion-gender-and-ethnicity-guesser-using-CNN-and-ML/blob/master/support_vector_classifier%20smile%20detection/a%20work%20of%20art.ipynb

I then tried to make a model for the emotion guesser, again using the jupyter notebook:
https://github.com/ashi72/age-emotion-gender-and-ethnicity-guesser-using-CNN-and-ML/blob/master/cnns_training/agr_cnn_try_1.ipynb

Here I made a confusion matrix with that model:
https://github.com/ashi72/age-emotion-gender-and-ethnicity-guesser-using-CNN-and-ML/blob/master/cnns_training/cnn_evaluations.ipynb

Lastly, I used a .py file to recall the pickled model and applied it to the webcam stream:
https://github.com/ashi72/age-emotion-gender-and-ethnicity-guesser-using-CNN-and-ML/blob/master/testing%20models%20with%20webcam/homestretch_3.2.py

There's a lot of bad coding practices, including organization and not creating testing code, but also in hindsight
there are a lot of things I regret not doing, such as not using GPU to train to speed up the process, and also I didn’t
save the trained CNN model after each epoch, so I was blindly guessing how many epochs would be best. I also failed at
making a good graph of validation accuracy to find the point where overtraining decreases the model’s accuracy. 
