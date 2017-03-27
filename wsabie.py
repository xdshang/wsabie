from itertools import cycle
import numpy as np
from scipy import sparse
import h5py
from evaluation import load_nuswide, normalize

rng = np.random.RandomState(1701)
transformer = []
batch_size = 100

# def load(data_file, label_file):
#   '''
#   data: hdf5 file
#   label_file: text file in COOR format for sparse matrix
#   '''
#   h5f = h5py.File(data_file, 'r')
#   dset = h5f['/data']
#   with dset.astype('float32'):
#     data = dset[:]

#   i = []
#   j = []
#   v = []
#   n = m = 0
#   with open(label_file, 'r') as fin:
#     for line in fin:
#       line = line.split()
#       n = max(n, int(line[0]))
#       m = max(m, int(line[1]))
#       i.append(int(line[0]) - 1)
#       j.append(int(line[1]) - 1)
#       v.append(1) # value should be binary
#   label = sparse.coo_matrix((v, (i, j)), shape = (n, m), 
#       dtype = np.dtype('b')).tolil()
#   data = data.toarray()
#   data = normalize(data, axis = 1)
#   label = label.tolil()

#   return data, label

def load():
  _, label, _, _, data = load_nuswide('nuswide.npz', 'train')
  data = data.toarray()
  data = normalize(data, axis = 1)
  label = label.tolil()

  return data, label

def save(fname, I, W):
  h5out = h5py.File(fname + '.h5', 'w')
  Iset = h5out.create_dataset('I', data = I)
  Wset = h5out.create_dataset('W', data = W)
  h5out.close()

def projection(I, W):
  norm_I = np.linalg.norm(I, axis = 1)
  norm_W = np.linalg.norm(W, axis = 1)
  for i in range(len(norm_I)):
    if norm_I[i] > 1:
      I[i, :] *= 1. / norm_I[i]
  for i in range(len(norm_W)):
    if norm_W[i] > 1:
      W[i, :] *= 1. / norm_W[i]

  return I, W

def train(I, W, data, label, lr = 0.001, maxIter = None):
  it = 0
  loss = 0

  sampleIter = cycle(rng.permutation(label.shape[0]))
  universe = set(range(label.shape[1]))

  I, W = projection(I, W)

  while True:
    # update
    sampleId = sampleIter.next()
    feat = np.dot(data[sampleId], I)
    # obtain label and vlabel (violate label)
    l = label.rows[sampleId]
    if len(l) == 0:
      continue
    vl = list(universe.difference(l))
    vllen = len(vl)

    delta_feat = np.zeros(feat.shape)
    delta_W = np.zeros(W.shape)
    for y in l:
      score = np.dot(W[y, :], feat)
      margin = -1
      esN = 0
      while margin <= 0 and esN < (vllen - 1):
        vy = vl[rng.randint(vllen)]
        vscore = np.dot(W[vy, :], feat)
        margin = vscore - score + 1
        esN += 1
      if margin > 0:
        rank = transformer[(vllen - 1) / esN]
        loss += rank * margin
        # gradient
        delta_feat += (W[y, :] - W[vy, :]) * rank
        temp = feat * rank
        delta_W[y, :] += temp 
        delta_W[vy, :] -= temp 

    I += np.tensordot(data[sampleId], delta_feat, axes = 0) * (lr / len(l))
    W += delta_W * (lr / len(l))
    I, W = projection(I, W) 
 
    it += 1
    if maxIter is not None and it == maxIter:
      print 'Finished training at iteration ', it, ' with loss: ', loss / ((it - 1) % batch_size + 1)
      break
    # print
    if it % batch_size == 0:
      print '\titer: ', it, '\tloss: ', loss / batch_size 
      loss = 0
    # save
    if it % label.shape[0] == 0:
      print 'saving model...'
      save('models/wsabie_model_iter_%d' % (it,), I, W)

  return I, W

if __name__ == '__main__':
  embed_dim = 300
  # load data
  # data, label = load('labels/sbu_decaf_norm_label.h5', 'labels/sbu_label.txt')
  data, label = load()
  print 'Data shape: ', data.shape
  print 'Label shape: ', label.shape
  # initialize transformer
  transformer = [0] * (label.shape[1] + 1)
  for i in range(label.shape[1]):
    transformer[i + 1] = transformer[i] + 1. / (i + 1)
  # initialize model
  I = rng.rand(data.shape[1], embed_dim).astype(data.dtype)
  W = rng.rand(label.shape[1], embed_dim).astype(data.dtype)
  # train loop
  I, W = train(I, W, data, label, maxIter = 2 * data.shape[0])
  # save to hdf5 file
  save('models/wsabie_model', I, W)
