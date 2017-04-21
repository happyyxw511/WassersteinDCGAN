import sys
import os
import pickle
import tarfile
import subprocess
import numpy as np

# ----------------------------------------------------------------------------
def _load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
      datadict = pickle.load(f)
      X = datadict['data']
      Y = datadict['labels']
      X = X.reshape(10000, 3, 32, 32).astype("float32")
      Y = np.array(Y, dtype=np.uint8)
      return X, Y

def load_mnist():
  # We first define a download function, supporting both Python 2 and 3.
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print "Downloading %s" % filename
    urlretrieve(source + filename, filename)

  file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
  for file_name in file_names:
    print file_name
    if not os.path.exists(file_name):
      download(file_name)
    out_path = os.path.join('./',file_name)
    cmd = ['gzip', '-d', out_path]
    print('Decompressing ', file_name)
    subprocess.call(cmd)
  data_dir = './'

  fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
  loaded = np.fromfile(file=fd,dtype=np.uint8)
  trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

  fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
  loaded = np.fromfile(file=fd,dtype=np.uint8)
  trY = loaded[8:].reshape((60000)).astype(np.float)

  fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
  loaded = np.fromfile(file=fd,dtype=np.uint8)
  teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

  fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
  loaded = np.fromfile(file=fd,dtype=np.uint8)
  teY = loaded[8:].reshape((10000)).astype(np.float)

  trY = np.asarray(trY)
  teY = np.asarray(teY)

  X = np.concatenate((trX, teX), axis=0)
  y = np.concatenate((trY, teY), axis=0).astype(np.int)

  seed = 547
  np.random.seed(seed)
  np.random.shuffle(X)
  np.random.seed(seed)
  np.random.shuffle(y)

  return X/255.


def load_cifar10(dest_directory='.'):
  """Download and extract the tarball from Alex's website."""
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    if sys.version_info[0] == 2:
      from urllib import urlretrieve
    else:
      from urllib.request import urlretrieve

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
  xs, ys = [], []
  for b in xrange(1, 6):
    f = 'cifar-10-batches-py/data_batch_%d' % b
    X, Y = _load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  Xte, Yte = _load_CIFAR_batch('cifar-10-batches-py/test_batch')
  return Xtr, Ytr, Xte, Yte
