import os
import numpy as np
import sklearn.model_selection
from sklearn.metrics import accuracy_score
import torch
import argparse
from joblib import dump, load

from sklearn import metrics

from sklearn import svm

def get_parser():
    parser = argparse.ArgumentParser(description='train a classifier to recognize trojan models')
    parser.add_argument('-f', '--filename', help='name of the pt file that stores the training data', default='separate_features.pt')
    parser.add_argument('--split', dest='split', action='store_true')
    parser.add_argument('--no-split', dest='split', action='store_false')
    parser.set_defaults(split=True)
    
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    datafile = args.filename
    data = torch.load(datafile)['table_ann']

    x = torch.stack(data['fvs'], dim=0).detach().cpu().numpy()
    print(x.shape)
    y = np.asarray(data['label'])

    clf = svm.SVC(kernel='linear')
    if args.split:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=42)
        print('x_train', x_train.shape)
        print('x_test', x_test.shape)
        clf.fit(x_train, y_train)
        y_preds = clf.predict(x_train)
        print(y_preds)
        print('train acc', accuracy_score(y_train.reshape(-1), np.asarray(y_preds)))
        y_preds = clf.predict(x_test)
        print('test acc', accuracy_score(y_test.reshape(-1), np.asarray(y_preds)))
    else:
        clf.fit(x, y)
        # train accuracy
        y_preds = clf.predict(x)
        print(y_preds)
        print('train acc', accuracy_score(y.reshape(-1), np.asarray(y_preds)))

    dump(clf, 'svm_norm_features.joblib') 
