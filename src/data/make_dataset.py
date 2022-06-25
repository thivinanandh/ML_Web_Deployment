# -*- coding: utf-8 -*-
from importlib.resources import path
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import tensorflow as tf
from references.data_dictionary import ModelParameters
import os
import pickle

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    ## read the data from the raw folder and save them into the Processed Folder. 
    ## The Processed Folder should have both the train and test data split and saved as a pickle file 
    pathval = os.path.join(os.getcwd(),input_filepath,ModelParameters["RawDataName"])
    print("PATH : ", pathval)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
                path=os.path.join(os.getcwd(),input_filepath,ModelParameters["RawDataName"]))


    logger = logging.getLogger(__name__)
    logger.info('Data set Created from the raw Data')



    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    logger.info('Validated the authenticity of the model data')

    

    ## Dump the files as in the extracted section 
    with open(f"{output_filepath}/x_test",'wb') as f:
        pickle.dump(x_test,f)
    
    with open(f"{output_filepath}/x_train",'wb') as f:
        pickle.dump(x_train,f)
    
    with open(f"{output_filepath}/y_test",'wb') as f:
        pickle.dump(y_test,f)
    
    with open(f"{output_filepath}/y_train",'wb') as f:
        pickle.dump(y_train,f)


    logger.info(f'Saved all the processed data into {output_filepath} folder')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
