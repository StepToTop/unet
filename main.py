from model import *
from data import *

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'data/train', 'image', 'label', data_gen_args)
for i in myGene:
    print(i)
    exit()
exit()
model = res_attention_Unet()
exit()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=1, epochs=1)
exit()
testGene = testGenerator("data/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/membrane/test", results)
