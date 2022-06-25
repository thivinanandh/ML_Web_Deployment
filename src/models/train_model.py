import pickle
from references.data_dictionary import ModelParameters
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import json

def train_model():
    """ 
    Trains the model based on the parameters provided on the data_dictionary model
    """
    ## obtain the loaded data 
    with open('./data/processed/x_test' , 'rb') as f:
        x_test = pickle.load(f)
    with open('./data/processed/x_train' , 'rb') as f:
        x_train = pickle.load(f)
    with open('./data/processed/y_test' , 'rb') as f:
        y_test = pickle.load(f)
    with open('./data/processed/y_train' , 'rb') as f:
        y_train = pickle.load(f)
    
    ## normalise data
    x_train = x_train/255.0
    x_test = x_test/255.0

    LayersArray  = [];
    
    dataAugumentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.95),
        # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(1,28,28)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.95),
        # tf.keras.layers.experimental.preprocessing.RandomTranslation(0.3,0.2),
        # tf.keras.layers.experimental.preprocessing.RandomHeight(0.1)
    ])

    convolution = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),  
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),  
        tf.keras.layers.Flatten()
    ])

    # LayersArray.append(dataAugumentation);
    # LayersArray.append(tf.keras.layers.Flatten(input_shape=(28, 28)))

    LayersArray.append(convolution)

    for i in range(ModelParameters['Layers']['numHiddenLayers']):
        if(ModelParameters['Layers'][i]['type'] =="Dense"):
            LayersArray.append(tf.keras.layers.Dense(ModelParameters['Layers'][i]['neurons'],activation=ModelParameters['Layers'][i]['activation']))

    LayersArray.append(tf.keras.layers.Dense(10))

    model = tf.keras.models.Sequential(LayersArray)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(ModelParameters['Optimizers']['LR']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


    model.fit(
        x_train,y_train,
        epochs=6,
        validation_data=(x_test,y_test)
    )

    loss, accuracy = model.evaluate(x_test,y_test,verbose=0)
    
    with open("reports/metrics.json", 'w') as outfile:
        json.dump({ "accuracy": accuracy, "loss": loss}, outfile)

    plt.bar(["Accuracy","Loss"],[accuracy,loss])
    plt.title("Error Metrics")
    plt.savefig("reports/metrics.png")

    logger = logging.getLogger(__name__)
    logger.info('model Generated.')
    logger.info(f'Loss : {loss} , Accuracy : {accuracy}')



    model.save("./models/model1")
    
    logger = logging.getLogger(__name__)
    logger.info('Model saved in the folder')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_model();