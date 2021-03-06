import sys,os
import random 
import numpy
from descriptors import * 
from musicfeatures import Features, Num, normalize

CONFIG = {'model': 'model_allb'}# model_tri1
params = {'n_fft':4096, 'hop_len':64, 'func': np.mean}
emotions = ['anger', 'happy', 'relax', 'sad']

def depickle(model_name):
    """
    From given pickle file it receives info about:
    classifier, normalization params, features info, labels coding 
    """
    import cPickle

    with open('%s.pkl'%model_name,'rb') as f:
        model = cPickle.load(f)

    clf = model['classifier']
    normin = model['norm']['min']
    normax = model['norm']['max']
    featinfo = model['featinfo']
    coding = model['coding']
    return clf, normin, normax, featinfo, coding

def calculate_features(path,piece_len=30):
    """
    It return features and unknown class 'x' for music piece from *path*
    Set of features:  rms, hoc, beats, chromagram, tempo, spectral centroids
    """
    try:
        musicfeat = Features(path,'x',piece_len=piece_len)
        params.update({'fs': musicfeat.sr})
        musicfeat.windowing(10,1) 
        musicfeat.add_winbased_features(rms)
        musicfeat.add_winbased_features(simple_hoc)
        musicfeat.add_winbased_features(beats,params)
        musicfeat.add_winbased_features(chromagram_feat, params)
        musicfeat.add_winbased_features(tempo,params)
        musicfeat.add_winbased_features(spectral_centroids,params)
        feats,clas = musicfeat.example
    except Exception as e:
        print e
        feats,clas = 0,0
    return feats, clas


class MusicEmoReco():
    "Main application for music emotion recognition"
    def __init__(self,path, parent=None):
        # super(MusicEmoReco, self).__init__(parent)
        self.path = path
        self.load_model()
        self.initPlot()

    def initPlot(self):
        self.make_features()
    
    def plot_reco(self,predictions):
        predictions*=100
        print("Predictions : ")
        # print(predictions)
        for i in range(len(emotions)) :
            print(emotions[i], predictions[i])

    def load_model(self):
        "It loads a file with model saved as a dictionary in python cPickle"
        try:
            self.clf, self.normin, self.normax, featinfo, coding = depickle(CONFIG['model'])
        except Exception as e:
            raise e

    def make_features(self):
        print("Calculating features for classification")
        self.feats,self.clas = calculate_features(self.path)
        self.classify()

   

    def classify(self):
        "If features was calculated it plots bars, otherwise it returns error window"
        if type(self.feats)!=int and self.clas!=0:
            self.feats = (self.feats - self.normin)/(self.normax - self.normin)
            self.plot_reco(self.clf.predict_proba(self.feats)[0])
        else:
            self.ups('Bad file format!')
            print('Bad file format!')

if __name__ == '__main__':

    song_name = sys.argv[1:][0]
    folder_path = os.path.join(os.getcwd(),'audio_files')
    #path contains the path of the audio file to be analyzed
    # path = "/home/vagisha/Projects/Django/musicemotionrecognition/audio_files/song.mp3"
    path = os.path.join(folder_path,song_name)
    print(path)
    MusicEmoReco(path)