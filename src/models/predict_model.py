import pickle
from references.data_dictionary import ModelParameters
import logging
import tensorflow as tf
import numpy as np
import cv2
import numpy as np

def image2numpy():
    img = cv2.imread('./data/processed/3.png')
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray_image, dsize=(28, 28))
    # cv2.imwrite("1_post.png", gray_image)
    # cv2.imwrite("1_post2.png", res)
    return res;

def predict_model(modelNo,data,isImage):
    """ 
    Predicts the number based on the Image
    """
    if(modelNo == 1):
        ## load model 1
        Savedmodel = tf.keras.models.load_model('./models/model1');
    
    logger = logging.getLogger(__name__)
    logger.info('model Loaded.')

    if(isImage==True):
        predData =  image2numpy()
    else:
        predData = data

    ## predict data 
    pred = Savedmodel.predict(predData.reshape((1,28,28)));
    print(pred)
    number = np.argmax(pred)

    logger.info(f'Predicted Value is {number}')
    return number


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    with open('./data/processed/x_test' , 'rb') as f:
        x_test = pickle.load(f)
    with open('./data/processed/y_test' , 'rb') as f:
        y_test = pickle.load(f)
    logger.info(f'Actual Value is {np.argmin(y_test)}')
    
    predict_model(1,x_test[234],True);
    