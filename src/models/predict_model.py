import pickle
from references.data_dictionary import ModelParameters
import logging
import tensorflow as tf
import numpy as np
# import cv2
import numpy as np
from PIL import Image




def predict_model(modelNo,data):
    """ 
    Predicts the number based on the Image 
    """
    if(modelNo == 1):
        ## load model 1
        Savedmodel = tf.keras.models.load_model('./models/model1');
    
    logger = logging.getLogger(__name__)
    logger.info('model Loaded.')

    # im = Image.fromarray(data)
    # im.save("1_post.png")

    ## predict data 
    pred = Savedmodel.predict(data.reshape((1,28,28)));
    number = np.argmax(pred)
    logger.info(f'Predicted Value is {number}')

    return np.argmax(pred)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    with open('./data/processed/x_test' , 'rb') as f:
        x_test = pickle.load(f)
    with open('./data/processed/y_test' , 'rb') as f:
        y_test = pickle.load(f)
    logger.info(f'Actual Value is {np.argmin(y_test)}')
    
    # predict_model(1,x_test[234]);
    