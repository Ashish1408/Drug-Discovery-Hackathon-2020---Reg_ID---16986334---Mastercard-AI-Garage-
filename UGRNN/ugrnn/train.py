from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import os

import numpy as np
import pandas as pd

from input_data import DataSet
from ugrnn import UGRNN
from utils import model_params
np.set_printoptions(threshold=np.inf, precision=4)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
FLAGS = None


def main(*args):

    #Next 5 lines of code - To handle a Possible Error that can occur in UGRNN Code
    df_ext = pd.read_csv("../../External Test Set/External_Test_Set.csv")
    if (df_ext.shape[1] == 3):
        print("Moving Forward")
    else:
        df_ext.to_csv("../../External Test Set/External_Test_Set.csv")

    model_dir = os.path.join(FLAGS.output_dir, FLAGS.model_name)

#    if tf.io.gfile.exists(model_dir):
#        tf.io.gfile.DeleteRecursively(model_dir)
#    tf.io.gfile.makedirs(model_dir)

    with tf.Graph().as_default():

        sess = tf.Session()

        logp_col_name = FLAGS.logp_col if FLAGS.add_logp else None

        logger.info('Loading Training dataset from {:}'.format(FLAGS.training_file))
        train_dataset = DataSet(csv_file_path=FLAGS.training_file,
                                smile_col_name=FLAGS.smile_col,
                                target_col_name=FLAGS.target_col, 
                                logp_col_name=logp_col_name,
                                contract_rings=FLAGS.contract_rings)

        logger.info('Loading validation dataset from {:}'.format(FLAGS.validation_file))
        validation_dataset = DataSet(csv_file_path=FLAGS.validation_file, 
                                     smile_col_name=FLAGS.smile_col,
                                     target_col_name=FLAGS.target_col, 
                                     logp_col_name=logp_col_name,
                                     contract_rings=FLAGS.contract_rings)
        
        logger.info('Loading test dataset from {:}'.format(FLAGS.test_file))
        test_dataset = DataSet(csv_file_path=FLAGS.test_file, 
                               smile_col_name=FLAGS.smile_col,
                               target_col_name=FLAGS.target_col,
                               logp_col_name=logp_col_name,
                               contract_rings=FLAGS.contract_rings)
        logger.info("Creating Graph.")


        ugrnn_model = UGRNN(FLAGS.model_name, 
                            encoding_nn_hidden_size=FLAGS.model_params[0],
                            encoding_nn_output_size=FLAGS.model_params[1], 
                            output_nn_hidden_size  =FLAGS.model_params[2],
                            batch_size=FLAGS.batch_size, 
                            learning_rate=0.001, 
                            add_logp=FLAGS.add_logp, 
                            clip_gradients=FLAGS.clip_gradient)



        logger.info("Succesfully created graph.")

        init = tf.global_variables_initializer()
        sess.run(init)
        logger.info('Run the Op to initialize the variables')
        ugrnn_model.train(sess, FLAGS.max_epochs, train_dataset, validation_dataset, model_dir)
        print('Saving model...')
        ugrnn_model.save_model(sess, model_dir, FLAGS.max_epochs)
        
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#        hidden_train = ugrnn_model.Hidden(sess, train_dataset)
#        hidden_validate = ugrnn_model.Hidden(sess, validation_dataset)
        hidden_test = pd.DataFrame(ugrnn_model.Hidden(sess, test_dataset))
        Raw_Test_filtered = pd.read_csv("../../External Test Set/External_Test_Set_filtered.csv")
        hidden_test['Canonical SMILES'] = Raw_Test_filtered['Canonical SMILES']
        print('Hidden_test created!')
#        pd.DataFrame(hidden_train).to_csv("./data/DILI/Final_data/Predictions/train_HidenRepresentation.csv")
        hidden_test.to_csv("./data/DILI/Final_data/Predictions/UGRNN Encoddings.csv")
        
        # prediction_train = ugrnn_model.predict(sess, train_dataset)
        # prediction_validate = ugrnn_model.predict(sess, validation_dataset)
        # prediction_test = ugrnn_model.predict(sess, test_dataset)
    
        # save_results("./data/DILI/Final_data/Predictions/Combined_train_result.csv", train_dataset.labels, prediction_train)
        # save_results("./data/DILI/Final_data/Predictions/Combined_val_result.csv",   validation_dataset.labels, prediction_validate)
        # save_results("./data/DILI/Final_data/Predictions/Combined_test_result.csv", test_dataset.labels, prediction_test)
        
        

        
if __name__ == '__main__':
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='default_model',
                        help='Name of the model')

    parser.add_argument('--max_epochs', type=int, default=350,
                        help='Number of epochs to run trainer.')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')

    parser.add_argument('--model_params', help="Model Parameters", dest="model_params", type=model_params, 
                        default = '12,3,5')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')

    parser.add_argument('--output_dir', type=str, default= './data/DILI/Model_Train_DILI_Data', #'train',
                        help='Directory for storing the trained models')

    parser.add_argument('--training_file', type=str, default='./data/DILI/Final_data/Combined_Train.csv',
                        help='Path to the csv file containing training data set')

    parser.add_argument('--validation_file', type=str, default='data/DILI/Final_data/Combined_Val.csv',
                        help='Path to the csv file containing validation data set')
#    
    parser.add_argument('--test_file', type=str, default='../../External Test Set/External_Test_Set.csv', #Raw_Test,Combined_Test
                        help='Path to the csv file containing test data set')
#
    parser.add_argument('--smile_col', type=str, default='Canonical SMILES')

    parser.add_argument('--logp_col', type=str, default='logp')

    parser.add_argument('--target_col', type=str, default='Label')

    parser.add_argument('--contract_rings', dest='contract_rings',default = True)

    parser.add_argument('--add_logp', dest='add_logp', default = False)
    
    parser.add_argument('--clip_gradient', dest='clip_gradient', default=False)
    
        
    
    

    FLAGS = parser.parse_args()
    
    main()
#    tf.app.run(main=main)