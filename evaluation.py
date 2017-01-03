__author__ = 'Xindi Shang'

import numpy as np
from scipy import sparse
import h5py
from sklearn import metrics

def load_nuswide(fname, mode = None):
  data = np.load(fname)

  gnd = data['gnd']
  label = sparse.csc_matrix((data['label_data'], data['label_indices'], data['label_indptr']))
  idx_tr = data['idx_tr']
  concept_name = data['concept_name']
  tag_name = data['tag_name']
  feat = sparse.csc_matrix((data['decaf_data'], data['decaf_indices'], data['decaf_indptr']))
  print 'nuswide dataset loaded.'
  
  if mode == None:
    return gnd, label, idx_tr, concept_name, tag_name, feat
  else:
    if mode == 'train':
      gnd = gnd[idx_tr, :]
      label = label[idx_tr, :]
      feat = feat[idx_tr, :]
    elif mode == 'test':
      idx_te = np.logical_not(idx_tr)
      gnd = gnd[idx_te, :]
      label = label[idx_te, :]
      feat = feat[idx_te, :]
    else:
      print 'Unknown mode. train, test or None.'
    return gnd, label, concept_name, tag_name, feat

def normalize(data, axis = 0):
  nrm = np.linalg.norm(data, axis = axis)
  nrm = 1. / nrm
  return data * np.expand_dims(nrm, axis = axis)

if __name__ == '__main__':
  # load dataset
  gnd, _, concept_name, tag_name, feat = load_nuswide('nuswide.npz', 'test')
  with open('labels/sbu_label_name.txt', 'r') as fin:
    tag_name = []
    for line in fin:
      tag_name.append(line.strip())
  # load model
  h5f = h5py.File('wsabie_model.h5', 'r')
  I = h5f['/I'][:]
  W = h5f['/W'][:]
  h5f.close()
  # predict
  print 'predicting...'
  feat = feat.toarray()
  feat = normalize(feat, axis = 1)
  feat = feat.dot(I)
  feat = normalize(feat, axis = 1)
  W = normalize(W, axis = 1)
  # evaluate
  print 'evaluating...'
  ap = []
  auc = []
  for i in range(gnd.shape[1]):
    tag_id = next((ind for ind, ele in enumerate(tag_name) if ele == concept_name[i]), None)
    if tag_id == None:
      print 'Concept name: ', concept_name, ' cannot be found in tag'
      continue
    score = feat.dot(W[tag_id, :])
    ap.append(metrics.average_precision_score(gnd[:, i], score))
    auc.append(metrics.roc_auc_score(gnd[:, i], score))
  print 'mAP: ', np.mean(ap)
  print 'AUC: ', np.mean(auc)
