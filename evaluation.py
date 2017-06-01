import numpy as np
from scipy import sparse
import h5py
from sklearn import metrics
import argparse

def load_nuswide(feat_fname, mode = None):
  meta = np.load('nuswide-meta.npz')
  gnd = meta['gnd']
  tag = sparse.csc_matrix((meta['tag_data'], meta['tag_indices'], meta['tag_indptr']))
  idx_tr = meta['is_train']
  gnd_name = meta['gnd_name']
  tag_name = meta['tag_name']
  img_name = meta['img_name']
  print('nuswide metadata loaded.')
  feat = np.load(feat_fname)
  feat = sparse.csc_matrix((feat['data'], feat['indices'], feat['indptr']))
  print('nuswide features loaded.')
  
  if mode == None:
    return gnd, tag, idx_tr, gnd_name, tag_name, img_name, feat
  else:
    if mode == 'train':
      gnd = gnd[idx_tr, :]
      tag = tag[idx_tr, :]
      img_name = img_name[idx_tr]
      feat = feat[idx_tr, :]
    elif mode == 'test':
      idx_te = np.logical_not(idx_tr)
      gnd = gnd[idx_te, :]
      tag = tag[idx_te, :]
      img_name = img_name[idx_te]
      feat = feat[idx_te, :]
    else:
      print('Unknown mode. train, test or None.')
    return gnd, tag, gnd_name, tag_name, img_name, feat

def normalize(X, axis = 0):
  nrm = np.linalg.norm(X, axis = axis)
  nrm = 1. / nrm
  return X * np.expand_dims(nrm, axis = axis)

def build_tagid2gndid(tag_name, gnd_name):
  tagid2gndid = dict()
  gnd_name = list(gnd_name)
  for i, tag in enumerate(tag_name):
    try:
      gndid = gnd_name.index(tag)
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
  parser = argparse.ArgumentParser(description = 'Evaluate a given model.')
  parser.add_argument('model_fname', type = str,
                      help = 'File path of the model.')
  args = parser.parse_args()
  # load metaset
  gnd, tag, gnd_name, tag_name, img_name, feat = load_nuswide('nuswide-decaf.npz', 'test')
  tag = tag.toarray()
  tagid2gndid = build_tagid2gndid(tag_name, gnd_name)
  # load model
  print('loading model {}...'.format(args.model_fname))
  h5f = h5py.File(args.model_fname, 'r')
  I = h5f['/I'][:]
  W = h5f['/W'][:]
  h5f.close()
  # predict
  print('predicting...')
  feat = feat.toarray()
  feat = normalize(feat, axis = 1)
  feat = feat.dot(I)
  # evaluate
  print('evaluating...')
  p5 = []
  p10 = []
  p20 = []
  ap = []
  for i in range(feat.shape[0]):
    score = W.dot(feat[i])
    # score = np.random.rand(tag_name.shape[0])
    rank_list = np.argsort(score)
    pred = np.zeros((gnd_name.shape[0],), dtype = np.int)
    set_pred(pred, rank_list[-5:], tagid2gndid)
    p5.append(metrics.precision_score(gnd[i], pred))
    set_pred(pred, rank_list[-10:], tagid2gndid)
    p10.append(metrics.precision_score(gnd[i], pred))
    set_pred(pred, rank_list[-20:], tagid2gndid)
    p20.append(metrics.precision_score(gnd[i], pred))
    if tag[i].sum() > 0:
      ap.append(metrics.average_precision_score(tag[i], score))
    if i % 10000 == 0:
      print(i)
  print('P@5: {}'.format(np.mean(p5)))
  print('P@10: {}'.format(np.mean(p10)))
  print('P@20: {}'.format(np.mean(p20)))
  print('mAP: {}'.format(np.mean(ap)))
  with open('precision10.txt', 'w') as fout:
    for i, x in enumerate(p10):
      fout.write('{} : {}\n'.format(img_name[i], x))
