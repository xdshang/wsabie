# Wsabie

Python implementation of Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie: Scaling up to large vocabulary image annotation." (2011).

#### Data format

Groundtruth, tags and meta data of NUS-WIDE dataset is in `nuswide_meta.npz`.

>Chua, Tat-Seng, et al. "NUS-WIDE: a real-world web image database from National University of Singapore." Proceedings of the ACM international conference on image and video retrieval. ACM, 2009.


You should add your own image features, and load them in `evaluation.py`, like:

`feat = data['decaf_feat']`,

if your features are sparse, then

`feat = sparse.csc_matrix((data['decaf_data'], data['decaf_indices'], data['decaf_indptr']))`.
