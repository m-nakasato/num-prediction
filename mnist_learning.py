import sys
print('モデルを選択してください')
print('1: 単純パーセプトロン')
print('2: 多層パーセプトロン')
print('3: CNN')
print('q: 終了')

mode = input('>>> ')

if mode == '1':
    base_name = 'mnist_model_simple'
    optimizer = 'rmsprop' #RMSPropオプティマイザ：lr(学習率)=0.001, rho=0.9, epsilon(微小量)=None(K.epsilon()), decay(各更新の学習率減衰)=0.0
elif mode == '2':
    base_name = 'mnist_model_multilayer'
    optimizer = 'rmsprop'
elif mode == '3':
    base_name = 'mnist_model_cnn'
    optimizer = 'adadelta'
else:
    sys.exit()

from datetime import datetime
start = datetime.now()

from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D #CNN用
from keras.optimizers import RMSprop

from keras import backend as K

import matplotlib.pyplot as plt

#MNISTデータの読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data() #train：訓練用、test：検証用、x：画像データ、y：正解ラベル

#画像データの加工
if mode != '3':
    #２次元（28x28）から１次元（784）に変換
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    input_shape = (784,)
else:
#image_data_format = channels_last（channels_firstの場合フォーマットが異なるので注意→keras.json参照）
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) #(60000, 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) #(10000, 28, 28, 1)
    input_shape = (28, 28, 1)
#uint8からfloat32（単精度浮動小数点型）にキャスト
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#値を8bit（256階調）からパーセンテージ（0.0～1.0）に変換
x_train /= 255
x_test /= 255

#正解ラベルの加工
#one-hotエンコーディング
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#モデルの構築
model = Sequential()
#単純パーセプトロン
if mode == '1':
    model.add(Dense(10, activation='softmax', input_shape=input_shape)) #全結合層
#多層パーセプトロン
elif mode == '2':
    model.add(Dense(512, activation='relu', input_shape=input_shape))   #隠れ層（全結合層）、活性化関数：relu（0以下のとき0、1より大きいとき入力をそのまま出力）
    model.add(Dropout(0.2))                                        #ドロップする割合（入力直後は0.2程度、隠れ層は0.5程度が望ましいとも）
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))                     #出力層、10ユニット[0-9]を用意し、ソフトマックス関数で確率分布を正規化（0.0-1.0）
elif mode == '3':
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

#訓練過程の設定（最適化アルゴリズム、損失関数、評価関数）
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',                     #交差エントロピー：推定値と正解ラベルのズレを計算。多クラス分類で用いられる
              metrics=['accuracy']
)

#学習（訓練）
history = model.fit(x_train,
                    y_train,
                    batch_size=128,                                #ミニバッチのサイズ（設定したサンプル数ごとに勾配の更新を行）
#                    epochs=2,
                    epochs=20,                                     #
                    verbose=1,
                    validation_data=(x_test, y_test))
import json
with open(base_name + '_history.json', mode='w') as f:
    f.write(json.dumps(history.history))


epochs = list(range(1, len(history.history['acc']) + 1))

#正解率
plt.plot(epochs, history.history['acc'], label='Training')
plt.plot(epochs, history.history['val_acc'], label='Validation')
plt.title('Model accuracy(' + base_name + ')')
plt.xlabel('Epoch num.')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend()
plt.savefig(base_name + '_acc.png')

plt.cla()
plt.clf()

#ロス率
plt.plot(epochs, history.history['loss'], label='Training')
plt.plot(epochs, history.history['val_loss'], label='Validation')
plt.title('Model loss(' + base_name + ')')
plt.xlabel('Epoch num.')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.savefig(base_name + '_loss.png')


#評価
score = model.evaluate(x_test, y_test, verbose=1)

#model.summary()

#モデルのグラフ構造をプロット
plot_model(model, to_file=base_name + '.png', show_shapes=True)

#モデルの保存
model.save(base_name + '.h5')
#モデルのみ
model_json = model.to_json()
model_yaml = model.to_yaml()
filePath = base_name + '.json'
filePath = base_name + '.yaml'
with open(filePath, mode='w') as f:
    f.write(model_json)
    f.write(model_yaml)
#Parameter（重み）のみ
model.save_weights(base_name + '_weights.hd5')

end = datetime.now()
elapsed = end - start

with open(base_name + '_evaluate.txt', mode='w') as f:
    f.write('Start: ' + start.isoformat() + '\n')
    f.write('End: ' + end.isoformat() + '\n')
    f.write('Elapsed: ' + str(elapsed) + '\n')
    f.write('Test loss: ' + str(score[0]) + '\n')
    f.write('Test accuracy: ' + str(score[1]) + '\n')

