import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D,LeakyReLU,Dense,Reshape,Flatten,Dropout,BatchNormalization,Conv2DTranspose,MaxPool2D
from tensorflow.keras.activations  import sigmoid
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
import argparse, json

import subprocess
cmd = "pip install matplotlib"
subprocess.run(cmd.split())
from matplotlib import pyplot as plt

class dcgan:
    def __init__(
        self,
        pix_num=28,
        latent_dim = (7,7),
        batch_size=32,
        epochs=20,
        generated_image_path='./gen_image/',
        dis_lr=1e-5,
        dis_bata_1=0.1,
        dc_lr = 2e-4,
        dc_beta_1=0.5
    ):
        self.latent_dim = latent_dim
        self.pix_num = pix_num # 28 x 28
        self.batch_size = batch_size
        self.epochs = epochs
        self.generated_image_path = generated_image_path
        
        # discriminator
        self.dis_model = self._discriminator()
        self.dis_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=dis_lr, beta_1=dis_bata_1))
        self.dis_model.trainable = False # combine_model を学習する際は discriminator の重みを固定
        
        # generator
        self.gen_model = self._generator()
        
        # combine
        self.combine_model = Sequential([self.gen_model, self.dis_model])
        self.combine_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=dc_lr, beta_1=dc_beta_1))

    def _generator(self):
        inputs = Input(shape=(7,7,1))
        x = Conv2D(128, (3,3),padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2DTranspose(128, (2,2), strides=(2,2))(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(64, (3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2DTranspose(64, (2,2), strides=(2,2))(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(1, (3,3), padding='same',activation='tanh')(x)
        model = Model(inputs=inputs, outputs=x)
        print('generator summary')
        model.summary()
        return model

    def _discriminator(self):
        inputs = Input(shape=(28,28,1))
        x = Conv2D(64, (3,3),padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPool2D(pool_size=(2, 2))(x) # 14x14
        x = Conv2D(64, (3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPool2D(pool_size=(2, 2))(x) # 7x7
        x = Conv2D(128, (3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        print('discriminator summary')
        model.summary()
        return model

    def _save_image(self,mini_X,epoch,idx):
        img_array = (mini_X.astype('float32') * 127.5 + 127.5)[:,:,:,0]
        plt.figure(figsize=(8, 4))
        plt.suptitle(f'epoch= {epoch},idx={idx}', fontsize=20)
        for i in range(self.batch_size):
            plt.subplot(4, 8, i + 1)
            plt.imshow(img_array[i])
            plt.gray()
            plt.xticks([])
            plt.yticks([])
        if not os.path.exists(self.generated_image_path):
            os.mkdir(self.generated_image_path)
        filename = f'fassion_mnist_{str(epoch).zfill(4)}_{str(idx).zfill(4)}.png'
        filepath = os.path.join(self.generated_image_path, filename)
        plt.savefig(filepath)
        
    def train(self,X):
        if X.shape[0]%self.batch_size==0:
            batch_num = int(X.shape[0] / self.batch_size)
        else:
            batch_num = int(X.shape[0] / self.batch_size) + 1
        print(f'Number of batches: {batch_num}')
        
        for epoch in range(self.epochs):
            for idx in range(batch_num):
                tmp_batch_size = self.batch_size if idx != batch_num -1 or X.shape[0]%self.batch_size == 0 else X.shape[0]%self.batch_size
                noise = np.random.uniform(-1, 1, (tmp_batch_size,self.latent_dim[0],self.latent_dim[1],1))
                image_batch = X[idx*self.batch_size:idx*self.batch_size+tmp_batch_size]
                generated_images = self.gen_model.predict(noise, verbose=0)
                if idx % 100 == 0:
                    self._save_image(generated_images,epoch,idx)
                
                # discriminator
                mini_X = np.concatenate((image_batch, generated_images))
#                 mini_y = [1]*tmp_batch_size + [0]*tmp_batch_size
                mini_y = np.concatenate((np.ones(tmp_batch_size),np.zeros(tmp_batch_size)))
                d_loss = self.dis_model.train_on_batch(mini_X, mini_y)

                # generator
                noise = np.random.uniform(-1, 1, (self.batch_size,self.latent_dim[0],self.latent_dim[1],1))
#                 g_loss = self.combine_model.train_on_batch(noise, [1]*self.batch_size)
                g_loss = self.combine_model.train_on_batch(noise, np.ones(self.batch_size))
                print(f"epoch: {epoch} batch: {idx}, g_loss: {g_loss}, d_loss: {d_loss}")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--intermediate', type=str, default=os.environ['SM_OUTPUT_INTERMEDIATE_DIR'])

    return parser.parse_known_args()

def load_training_data(base_dir):
    X = np.load(os.path.join(base_dir, 'train_X.npy'))
    return X

if __name__ == "__main__":
    args, unknown = _parse_args()
    train_X = load_training_data(args.train)
    
    DCGAN = dcgan(
        epochs=args.epochs,
        generated_image_path=args.intermediate
    )
    DCGAN.train(train_X)
    
    save_model_path = os.path.join(args.sm_model_dir, '1.h5')
    DCGAN.gen_model.save(save_model_path)
    exit()