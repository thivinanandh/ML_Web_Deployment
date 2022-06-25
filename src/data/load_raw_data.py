## Fetch the MNIST data from the URL and save it in the Raw format
import urllib.request



from references.data_dictionary import ProjectParameters
urllib.request.urlretrieve(ProjectParameters["RawDataURL"], "./data/raw/minist.npz")

