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

def build_tagid2gndid(tag_name, concept_name):
  tagid2gndid = dict()
  concept_name = list(concept_name)
  for i, tag in enumerate(tag_name):
    try:
      gndid = concept_name.index(tag)
      tagid2gndid[i] = gndid
    except ValueError:
      pass
  return tagid2gndid

def set_pred(pred, indices, tagid2gndid):
  for ind in indices:
    try:
      pred[tagid2gndid[ind]] = 1
    except KeyError:
      pass

if __name__ == '__main__':
  # load dataset
  gnd, tag, concept_name, tag_name, feat = load_nuswide('nuswide.npz', 'test')
  tag = np.array(tag.todense())
  tagid2gndid = build_tagid2gndid(tag_name, concept_name)
  # with open('labels/sbu_label_name.txt', 'r') as fin:
  #   tag_name = []
  #   for line in fin:
  #     tag_name.append(line.strip())
  # load model
  h5f = h5py.File('models/wsabie_model.h5', 'r')
  I = h5f['/I'][:]
  W = h5f['/W'][:]
  h5f.close()
  # predict
  print 'predicting...'
  feat = feat.toarray()
  feat = normalize(feat, axis = 1)
  feat = feat.dot(I)
  # feat = normalize(feat, axis = 1)
  # W = normalize(W, axis = 1)
  # evaluate
  print 'evaluating...'
  p5 = []
  p10 = []
  p20 = []
  ap = []
  for i in range(feat.shape[0]):
    score = W.dot(feat[i])
    # score = np.random.rand(tag_name.shape[0])
    rank_list = np.argsort(score)
    pred = np.zeros((concept_name.shape[0],), dtype = np.int)
    set_pred(pred, rank_list[-5:], tagid2gndid)
    p5.append(metrics.precision_score(gnd[i], pred))
    set_pred(pred, rank_list[-10:], tagid2gndid)
    p10.append(metrics.precision_score(gnd[i], pred))
    set_pred(pred, rank_list[-20:], tagid2gndid)
    p20.append(metrics.precision_score(gnd[i], pred))
    if tag[i].sum() > 0:
      ap.append(metrics.average_precision_score(tag[i], score))
    if i % 10000 == 0:
      print i
  # auc = []
  # for i in range(gnd.shape[1]):
  #   tag_id = next((ind for ind, ele in enumerate(tag_name) if ele == concept_name[i]), None)
  #   if tag_id == None:
  #     print 'Concept name: ', concept_name, ' cannot be found in tag'
  #     continue
  #   score = feat.dot(W[tag_id, :])
  #   ap.append(metrics.average_precision_score(gnd[:, i], score))
  #   auc.append(metrics.roc_auc_score(gnd[:, i], score))
  print 'P@5:', np.mean(p5)
  print 'P@10:', np.mean(p10)
  print 'P@20:', np.mean(p20)
  print 'mAP:', np.mean(ap)
  # print 'AUC: ', np.mean(auc)
