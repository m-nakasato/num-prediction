import sys
print('モデルを選択してください')
print('1: 単純パーセプトロン')
print('2: 多層パーセプトロン')
print('3: CNN')
print('q: 終了')

mode = input('>>> ')

if mode == '1':
    base_name = 'mnist_model_simple'
elif mode == '2':
    base_name = 'mnist_model_multilayer'
elif mode == '3':
    base_name = 'mnist_model_cnn'
else:
    sys.exit()

from keras.models import load_model
from keras.models import model_from_yaml
#from keras.preprocessing.image import img_to_array, load_img
import numpy
from PIL import Image, ImageOps

import time

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.virtual import viewport
from luma.core.legacy import text, show_message
from luma.core.legacy.font import proportional, CP437_FONT, TINY_FONT, SINCLAIR_FONT, LCD_FONT

#LEDマトリクス準備
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=0, rotate=0)
device.contrast(8)

#モデルの読み込み
model = load_model(base_name + '.h5')

#filePath = base_name + '.yaml'
#with open(filePath) as f:
#    model_yaml = f.read()

#print(model_yaml)

#model = model_from_yaml(model_yaml)

#model.load_weights(base_name + '_weights.hd5')

model.summary()



while 1:
#for i in range(10):

    #対象画像を読み込み
    image = Image.open('input.bmp')
#    image = Image.open('meiryo_digits/' + str(i) + '.bmp')
#    image = Image.open('handwritten_digits/' + str(i) + '.bmp')
    #ネガポジ反転
    image = ImageOps.invert(image)
    #グレースケール変換
    image = image.convert('L')
    #numpy配列として読込
    x = numpy.array(image)
#    print(x.dtype)
#    print(x[14][14])
    x = x.astype('float32')
#    x = x.astype('float64')
#    print(x.dtype)
#    x /= 255
    x = numpy.true_divide(x, 255)
#    print(x[14][14])

    if mode != '3':
        #１次元に変換
        x = x.reshape(1, -1)
    else:
#        x = x.reshape(28, 28, 1)
#        x = numpy.expand_dims(x, axis=0) #(1, 28, 28, 1)
        x = x.reshape(1, 28, 28, 1)
#        print(x.shape)

    #result = model.predict(x)
    result = model.predict_classes(x)
    print(result[0])

    #LEDマトリクス描画
    with canvas(device) as _canvas:
        text(_canvas, (0, 0), str(result[0]), fill="white")

    time.sleep(1)

