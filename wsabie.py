from itertools import cycle
import numpy as np
from scipy import sparse
import h5py
from evaluation import load_nuswide, normalize

rng = np.random.RandomState(1701)
transformer = []
batch_size = 100

def load():
  _, label, _, label_name, _, data = load_nuswide('nuswide-decaf.npz', 'train')
  data = data.toarray()
  data = normalize(data, axis = 1)
  label = label.tolil()
  return data, label, label_name

def save(fname, I, W):
  h5out = h5py.File(fname + '.h5', 'w')
  Iset = h5out.create_dataset('I', data = I)
  Wset = h5out.create_dataset('W', data = W)
  h5out.close()

def projection(X):
  norm_X = np.linalg.norm(X, axis = 1)
  for i in range(len(norm_X)):
    if norm_X[i] > 1:
      X[i, :] *= 1. / norm_X[i]
  return X

def initialize_word_embeddings(label_name, embed_dim):
  import gensim
  model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
  assert model.syn0.shape[1] == embed_dim
  W = []
  for name in label_name:
    W.append(model[name])
  return np.asarray(W)

def train(I, W, data, label, lr_I = 0.001, lr_W = 0.001, maxIter = None):
  it = 0
  loss = 0

  sampleIter = cycle(rng.permutation(label.shape[0]))
  universe = set(range(label.shape[1]))

  I = projection(I)
  W = projection(W)

  print('Start training with lr_I {}, lr_W {}, maxIter {}'.format(lr_I, lr_W, maxIter))

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

    I += np.tensordot(data[sampleId], delta_feat, axes = 0) * (lr_I / len(l))
    W += delta_W * (lr_W / len(l))
    if lr_I > 0.:
      I = projection(I)
    if lr_W > 0.:
      W = projection(W)
 
    it += 1
    if maxIter is not None and it == maxIter:
      print('Finished training at iteration {} with loss: {}'.format(it, loss / ((it - 1) % batch_size + 1)))
      break
    if it % batch_size == 0:
      print('\titer: {}\tloss: {}'.format(it, loss / batch_size))
      loss = 0
    # save
    if it % label.shape[0] == 0:
      print('saving model...')
      save('models/wsabie_model_iter_{}'.format(it), I, W)

  return I, W

if __name__ == '__main__':
  embed_dim = 300
  random_init_W = True
  # load data
  data, label, label_name = load()
  print('Data shape: {}'.format(data.shape))
  print('Label shape: {}'.format(label.shape))
  # initialize transformer
  transformer = [0] * (label.shape[1] + 1)
  for i in range(label.shape[1]):
    transformer[i + 1] = transformer[i] + 1. / (i + 1)
  # initialize model
  I = rng.rand(data.shape[1], embed_dim).astype(data.dtype)
  if random_init_W:
    W = rng.rand(label.shape[1], embed_dim).astype(data.dtype)
  else:
    W = initialize_word_embeddings(label_name, embed_dim)
  # train loop
  I, W = train(I, W, data, label, lr_I = 0.001, lr_W = 0.00001,
      maxIter = 2 * data.shape[0])
  # save to hdf5 file
  save('models/wsabie_model', I, W)
