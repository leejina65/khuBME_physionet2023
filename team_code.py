#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

from helper_code import *
import torch
import re
import natsort
import matplotlib.pyplot as plt
import torchvision
import tensorflow as tf
import numpy as np, os, sys
import mne
import joblib
import csv
import pywt
import scipy.io
from glob import glob
import multiprocessing as mp
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils.np_utils import to_categorical

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if verbose >= 1:
        print('Finding the Challenge data...')

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    ###
    # flist=fname(data_folder,patient_ids) #high quality signals only
    # process_files(data_folder, flist)
    # imgfolder(data_folder,flist)
    ###

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        current_features = get_features(patient_metadata, recording_metadata, recording_data)
        features.append(current_features)

        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Impute any missing features; use the mean value by default.
    input=natsort.natsorted(glob(data_folder+'/*.npy'))

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    feature_scl=scalercolumn(features)

    # Train the models.
    lr = 0.001
    batch_size = 1
    train_gen_cpc = generator(data_folder=data_folder, input=input,
                          outcome=outcomes,cpc=cpcs,feature=feature_scl,ttv='train',
                          batch_size=batch_size, is_train=True,flag='cpc')  # get batch size-input
    train_gen_outcome = generator(data_folder=data_folder, input=input,
                          outcome=outcomes,cpc=cpcs,feature=feature_scl,ttv='train',
                          batch_size=batch_size, is_train=True,flag='outcome')  # get batch size-input

    cpc_model = CpcModel()
    cpc_model.build(input_shape=[(None, 64, 128, 18), (None, 7)])  # set the input shapes manually
    cpc_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cpc_model.fit_generator(train_gen_cpc, steps_per_epoch=50,
                                  epochs=300, verbose=2, validation_steps=50)

    outcome_model=OutcomeModel()
    outcome_model.build(input_shape=[(None, 64, 128, 18), (None, 7)])  # set the input shapes manually
    outcome_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = outcome_model.fit_generator(train_gen_outcome, steps_per_epoch=50,
                                  epochs=300, verbose=2, validation_steps=50)

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
def fname(dfpath,patient_ids): #filtering high quality data
    fname={}
    for i in patient_ids:
      path0=dfpath+'/'+i
      path1=os.path.join(path0,i+'.tsv')
      lines=[]
      with open(path1) as f: #tsv파일 읽기/
        tr=csv.reader(f,delimiter='\t')
        for row in tr:
          lines.append(row)

      flist=[]
      for j in range(1,len(lines)):
        if lines[j][2]=='nan': continue
        if float(lines[j][2])>0.9:
          flist.append(lines[j][3])
      fname[i]=flist
    return fname

def process_files(dfpath,files):
    with mp.Pool() as pool:
        pool.starmap(cwt_gaus2,((dfpath,key, value) for key, values in files.items() for value in values))

def cwt_gaus2(dfpath,key,value):
    mat_file = scipy.io.loadmat(dfpath+'/'+key+'/'+value+'.mat')
    input=mat_file['val']
    for i in range(18):
        if not os.path.isfile(dfpath+'/'+key+'/'+value+'_'+str(i+1)+'_.png'):
            t=np.linspace(0,270,27000)
            signal=input[i]
            scales=np.arange(0.1,12) #no. of scales
            coef,freqs=pywt.cwt(signal,scales,'gaus2')

            plt.plot(figsize=(5, 2))
            plt.imshow(abs(coef),extent=[0,27000,12,0.1],interpolation='bilinear',cmap='bone',aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
            plt.gca().invert_yaxis()
            plt.yticks(np.arange(0.1,12,1))
            plt.xticks(np.arange(0,27000,3000))
            plt.axis('off')
            plt.savefig(dfpath+'/'+key+'/'+value+'_'+str(i+1)+'_.png',bbox_inches='tight',pad_inches=0)
            plt.close()

def imgfolder(imgroot,flist):  # Tensor => 18channel::: Save numpy.file
    trans = transforms.Compose([transforms.Resize((64, 128)),
                                transforms.Grayscale(1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

    pngs,channel_list,time_list,tensor_list,patient=[],[],[],[],[]
    #the list of every patient img files
    for key,value in flist.items():
        head=os.path.join(imgroot,key)
        pngs+=natsort.natsorted(glob(head+'/*.png'))

    idx=0
    for key in flist:
        for timer in range(len(flist[key])*18):
            with Image.open(pngs[idx]) as img:
                img_tf = trans(img)
                channel_list.append(img_tf)
                idx+=1

                if len(channel_list) == 18:
                    tensor = torch.cat(channel_list, dim=0).permute(1,2,0)
                    time_list.append(tensor)
                    channel_list=[]

        tensor_list = torch.stack(time_list, dim=0)
        npsave = np.asarray(tensor_list)
        np.save(imgroot+'/'+key+ '.npy', npsave)
        time_list,tensor_list=[],[]

def createFolder(directory): #이미 디렉토리를 만들었으면 무시
  try:
    if not os.path.exists(directory): os.makedirs(directory)
    return directory
  except OSError:
    print("There is already the directory:",directory)

def scalercolumn(feature): #nparray(N,8) age, female, male, other, rosc, ohca, vfib, ttm
    scaler=MinMaxScaler()
    scaler.fit(feature)
    scaled_feature=scaler.transform(feature)
    return scaled_feature

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

#model
class CpcModel(tf.keras.Model):
    def __init__(self):
        super(CpcModel, self).__init__()

        # Define the layers
        self.conv1 = layers.Conv2D(filters=16, kernel_size=(9,9), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.maxpool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv2 = layers.Conv2D(64, (9,9), activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.maxpool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv3 = layers.Conv2D(128, (9,9), activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')
        self.maxpool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(256)
        self.bn4 = layers.BatchNormalization()
        self.relu4 = layers.Activation('relu')

        self.fc2 = layers.Dense(128)
        self.bn5 = layers.BatchNormalization()
        self.relu5 = layers.Activation('relu')

        self.fc3 = layers.Dense(5, activation='softmax')

    def call(self, inputs):
        x, bin_input = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = layers.concatenate([x, bin_input], axis=1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)

        output = self.fc3(x)

        return output
class OutcomeModel(tf.keras.Model):
    def __init__(self):
        super(OutcomeModel, self).__init__()

        # Define the layers
        self.conv1 = layers.Conv2D(filters=16, kernel_size=(9,9), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.maxpool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv2 = layers.Conv2D(64, (9,9), activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.maxpool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv3 = layers.Conv2D(128, (9,9), activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')
        self.maxpool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(128)
        self.bn4 = layers.BatchNormalization()
        self.relu4 = layers.Activation('relu')

        self.fc2 = layers.Dense(64)
        self.bn5 = layers.BatchNormalization()
        self.relu5 = layers.Activation('relu')

        self.fc3 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, bin_input = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = layers.concatenate([x, bin_input], axis=1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)

        output = self.fc3(x)

        return output

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])


    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combine the features from the patient metadata and the recording data and metadata.
    #features = np.hstack((patient_features, recording_features))
    features=patient_features

    return features

class generator(Sequence):
    def __init__(self,data_folder,input,cpc,outcome,feature,ttv,batch_size,is_train,flag):
        self.is_train=is_train
        self.batch_size=batch_size
        self.ttv=ttv
        self.flag=flag

        path = os.path.join(data_folder, self.ttv)
        path_patient=find_data_folders(data_folder)

        inputs,cpcs,outcomes,features=[],[],[],[]
        for idx,i in enumerate(input):
            #input np -> vertically stacking
            temp=np.load(i)
            inputs.append(temp)

            #filling cpc,outcome,feature
            chsize=temp.shape[0] #(48,64,128,18)의 경우 채널 48개 -> feature이랑 batch모두 연장

            tempcpc=np.full((chsize),cpc[idx]-1,dtype=np.float16)
            tempoutcome = np.full((chsize), outcome[idx], dtype=np.float16)
            tempfeature = np.array([[f for f in feature[idx]] for i in range(chsize)])

            cpcs.append(tempcpc)
            outcomes.append(tempoutcome)
            features.append(tempfeature)

        self.x=np.vstack(inputs)
        self.cpc = np.hstack(cpcs)
        self.outcome = np.hstack(outcomes)
        self.bin_input = np.vstack(features)

        self.on_epoch_end()
        self.n_classes = len(np.unique(self.outcome))
        self.index = np.arange(len(self.x))
        if self.is_train:
            np.random.shuffle(self.index)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self,idx):
        batch_index = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.x[i] for i in batch_index]
        batch_bin_input = [self.bin_input[i] for i in batch_index]
        batch_cpc = [to_categorical(self.cpc[i],num_classes=5) for i in batch_index]
        batch_outcome = [self.outcome[i] for i in batch_index]

        if self.flag=='cpc':
            return (np.array(batch_x), np.array(batch_bin_input)), np.array(batch_cpc)
        elif self.flag=='outcome':
            return (np.array(batch_x), np.array(batch_bin_input)), np.array(batch_outcome)

    def on_epoch_end(self):
        self.index=np.arange(len(self.x))
        if self.is_train:
            np.random.shuffle(self.index)
