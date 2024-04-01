import configparser
from datasets import ECGSequence
from model import get_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping,
ReduceLROnPlateau)
from accuracy import CategoricalAccuracyPerClass

if __name__ == "__main__":
# 创建一个配置解析器对象
config = configparser.ConfigParser()

# 读取配置文件
config_path = r'./files/config.txt'
config.read(config_path)

args = {
'path_to_datas': config['config']['path_to_datas'],
'data_name': config['config']['data_name'],
'path_to_label': config['config']['path_to_label'],
'batch_size': int(config['config']['batch_size']),
'val_split': float(config['config']['val_split']),
'path_to_model': config['config']['path_to_model']
}

# 导入数据
trains, _ = ECGSequence.get_train_and_val(args['path_to_datas'], args['data_name'], args['path_to_label'],
args['batch_size'], args['val_split'])

# trains = trains.shuffle().repeat(70).batch_size(args['batch_size'])

print(trains.n_classes)
#
# # 导入模型
# model = get_model(trains.n_classes, (4096, 12))
#
# # 导入预训练模型
# pretrained_model = load_model(args['path_to_model'])
#
# # 模型参数预配置
# for i in range(4, 48):
# model.layers[i].set_weights(pretrained_model.layers[i].get_weights())
#
# for i in range(4, 48):
# model.layers[i].trainable = False
#
# # 模型配置
# loss = 'binary_crossentropy'
# lr = 0.001
# opt = Adam(lr)
#
# model.compile(loss=loss, optimizer=opt,
# metrics=[])
#
# # 回调函数配置
# callbacks = [TensorBoard(log_dir='./logs', write_graph=False),
# CSVLogger('files/training.log', append=False),
# ModelCheckpoint('files/backup_model_last_with_weights.hdf5'),
# # ModelCheckpoint('files/backup_model_best_with_weights.hdf5', save_best_only=True),
# EarlyStopping(patience=9, min_delta=0.00001),
# ReduceLROnPlateau(monitor='val_loss',
# factor=0.1,
# patience=7,
# min_lr=lr / 100),
# ]
#
# # 模型训练
# history = model.fit(trains,
# epochs=70,
# initial_epoch=0,
# callbacks=callbacks,
# # validation_data=vals,
# verbose=1)
#
# model.save("./files/final_model_with_weights/final_model_with_weights.hdf5")