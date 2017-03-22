import tensorflow as tf
import download
import os

# model download link
data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

# directory to download model
data_dir = 'vgg16/'

# model filename
path_graph_def = 'vgg16.tfmodel'

def maybe_download():
    """
    Download the VGG16 model from the internet if it does not already
    exist in the data_dir.
    """
    print("Downloading VGG16 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)

class VGG16:

	# Names for the convolutional layers in the model for use in Style Transfer.
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):

    	with tf.Graph().as_default() as self.graph:
    		# path to the model
    		path = os.path.join(data_dir, path_graph_def)

    		with tf.gfile.FastGFile(path, 'rb') as file:

    			# create empty graph-def
    			graph_def = tf.GraphDef()

    			# load proto-buf file into graph-def
    			graph_def.ParseFromString(file.read())

    			# import graph-def to current graph
    			tf.import_graph_def(graph_def, name='')

    		# assign the graph to a variable
    		self.graph = tf.get_default_graph()
            
    		self.input = self.graph.get_tensor_by_name('images:0')




    def get_output(self, layer_id):

    	return self.graph.get_tensor_by_name(self.layer_names[layer_id]+':0')


    def get_output_layerwise(self, layer_ids):

    	return [self.get_output(idx) for idx in layer_ids]


    def get_input_tensor(self):

    	return self.input