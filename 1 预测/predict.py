import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence
from tensorflow import keras
#
from myFunctionFolder.my_OS_Function import *

#hdf5模型中的导联数
n_link=6;
is_stand="_stand"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    # parser.add_argument('path_to_hdf5', type=str,
    #                     help='path to hdf5 file containing tracings')
    # parser.add_argument('path_to_model',  # or model_date_order.hdf5
    #                     help='file containing training model.')
    # parser.add_argument('--dataset_name', type=str, default='tracings',
    #                     help='name of the hdf5 dataset containing tracings')
    # parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
    #                     help='六类指标_12 csv file.')
    # parser.add_argument('-bs', type=int, default=32,
    #                     help='Batch size.')


    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    ##
    args.path_to_hdf5=r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_code\0 hdf5测试数据集修改\ecg_tracings_{}{}.hdf5".format(n_link,is_stand)
    args.dataset_name="tracings"
    args.path_to_model=r"D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\model\model.hdf5"
    # args.output_file= r".\predictOutput_{}.npy".format(n_link)
    args.output_file= r".\predictOutput_{}{}.npy".format(n_link,"_stand")
    args.bs=32
    ##

    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    # Import model
    model = load_model(args.path_to_model, compile=False)
    # model.compile(loss='binary_crossentropy', optimizer=Adam())
    # y_score = model.predict(seq,  verbose=1)
    #
    # # Generate dataframe
    # np.save(args.output_file, y_score)
    #
    # # print("Output predictions saved")
    #
    # # print(os.getcwd()+"\\"+args.output_file)
    #
    # print(np.load(os.getcwd()+"\\"+args.output_file).shape)

    keras.utils.plot_model(model, show_shapes=True)