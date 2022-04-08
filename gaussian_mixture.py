# A class to hold information on the GMM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

class GMM:

  def __init__(self, character):
    self.character = character
    self.k = self.get_bic()
    self.gmm = None

  # Can use this to get the Best BIC and therefore best K but was told on piazza to fix it.
  def get_bic(self):
    # After some testing the best response was 5 BIC
    return 5

  # Trains the data on a standard scalar set of the data
  def train(self, data):
    pipe = make_pipeline(StandardScaler(), GaussianMixture(n_components=self.k, covariance_type = 'tied'))
    self.gmm = pipe.fit(data)

  def weights(self):
    return self.gmm.weights_

  def predict(self, data):
    return self.gmm.score_samples(data)