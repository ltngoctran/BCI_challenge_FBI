import numpy as np                                                 # for dealing with data
from scipy.signal import butter, sosfiltfilt, sosfreqz  
import os
import pandas as pd
import pickle
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from ...config import data_dir, fit_dir, fit1_dir
# Step 1 Design a Butterworth Bandpass
def butter_bandpass_filter(raw_data, fs, lowcut = 1.0, highcut =40.0, order = 5):
    '''
    The filter is applied to the raw eeg data.
    :raw_data (nparray): data you want to process
    :fs (float): sampling rate
    :lowcut (float, optional): lowest frequency we will pass
    :highcut (float, optional): highest frequency we will pass
    :order (int, optional): order of filter
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog = False, btype = 'band', output = 'sos')
    filted_data = sosfiltfilt(sos, raw_data)
    return filted_data


def get_epoch_coefs(fs, epoch_s = 0, epoch_e = 700, bl_s = 0, bl_e = 100):
    '''
    epoch_s : epoch starting time relative to stmulus in miliseconds
    epoch_e : epoch ending time relative to stmulus in miliseconds
    bl_s    : baseline starting time relative to stmulus in miliseconds
    bl_e    : baseline ending time relative to stmulus in miliseconds
    '''
    # number of mark per epoch
    epoch_len = int((abs(epoch_e) - abs(epoch_s)) * (fs / 1000)) 
    e_s = int((epoch_s * (fs / 1000)))
    e_e = int((epoch_e * (fs / 1000)))
    b_s = int((abs(epoch_s) + bl_s) * (fs / 1000))
    b_e = int((abs(epoch_s) + bl_e) * (fs / 1000))

    return epoch_len, e_s, e_e, b_s, b_e




def generate_epoch(file_path,fs, channels, filter, baseline=True, epoch_s=0, epoch_e=700, bl_s=0, bl_e=100):
    data = pd.read_csv(file_path)
    data.loc[:, 'Time'] = data.loc[:, 'Time']*1000
    data['index'] = data.index.values
    mark_indices = np.asarray(data[data['FeedBackEvent'] == 1].index)
    e_len,e_s,e_e,b_s,b_e = get_epoch_coefs(fs,epoch_s,epoch_e,bl_s,bl_e)

    list_epoch  = []
    for channel in channels:
        epoch = np.zeros(shape=(int(mark_indices.shape[0]), e_len))
        raw_eeg = data[channel].values
        clean_eeg = filter(raw_eeg, fs, lowcut=1.0, highcut=40.0, order=5)
        for i, mark_idx in enumerate(mark_indices):
            epoch[i, :] = clean_eeg[mark_idx + e_s: mark_idx + e_e]
        if baseline:
            for i in range(0, int(mark_indices.shape[0])):
                epoch[i, :] = epoch[i, :] - np.mean(epoch[i, b_s:b_e])
    
        list_epoch.append(epoch)

    total_epoch=np.array(list_epoch).swapaxes(0,1)
    return total_epoch
def generate_combine_data(fs, channels, filter, baseline=True, epoch_s=0, epoch_e=700, bl_s=0, bl_e=100):
    # num_subj_train   = 16
    # num_subj_test    = 10
    num_session_per_subj = 5
    num_feedback_per_subj = 340
    num_subj = {'train':16,'test':10}
    arr_list = {'train': None, 'test':None}
    total_subj = {'train':None, 'test':None}
    for phase in ['train', 'test']:
        list_paths= sorted(os.listdir(data_dir+phase))
        arr_list[phase]= np.array(list_paths).reshape(num_subj[phase],num_session_per_subj)
        print(arr_list[phase])
    
    for phase in ['train', 'test']:
        print('phase',phase)
        list_total_subj= []
        for subj_id in range(num_subj[phase]):
            subj_dir = arr_list[phase][subj_id]
            print(subj_dir)
            list_subj_epoch = []
            for session_id in range(num_session_per_subj):
                session_dir = os.path.join( data_dir + phase ,subj_dir[session_id])
                data = generate_epoch(session_dir,fs,channels,filter,baseline,epoch_s,epoch_e,bl_s,bl_e)
                list_subj_epoch.append(data)
            subj_epoch = np.vstack(list_subj_epoch)
            print(subj_epoch.shape)
            list_total_subj.append(subj_epoch)
        total_subj[phase]= np.array(list_total_subj)
        print(total_subj[phase].shape)

    
    pickle.dump(total_subj, open(fit_dir+'total_subj', "wb" ))
    np.save(fit_dir+'train_data.npy', total_subj['train'],allow_pickle=True)
    np.save(fit_dir+'test_data.npy', total_subj['test'],allow_pickle=True)

def apply_pyriemann_data(train_data,test_data):
    XC = XdawnCovariances(nfilter=5) 
    TS = TangentSpace(metric='riemann')
    num_subj_train   = train_data.shape[0]
    num_subj_test    = test_data.shape[0]
    num_feedback_per_subj = train_data.shape[1]
    n_channels = train_data.shape[2]
    epoch_len  = train_data.shape[3]

    my_train_data = np.reshape(train_data, (num_subj_train * num_feedback_per_subj, n_channels, epoch_len))
    my_test_data  = np.reshape(test_data, (num_subj_test * num_feedback_per_subj, n_channels, epoch_len))
    y_train = pd.read_csv(data_dir+'TrainLabels.csv')['Prediction'].values
    # transform our data
    X_train = XC.fit_transform(my_train_data, y_train)
    X_train = TS.fit_transform(X_train)

    X_test = XC.transform(my_test_data)
    X_test = TS.transform(X_test)
    print(X_train.shape)
    print(X_test.shape)
    np.save(fit1_dir+'X_train.npy', X_train,allow_pickle=True)
    np.save(fit1_dir+'X_test.npy', X_test,allow_pickle=True)








