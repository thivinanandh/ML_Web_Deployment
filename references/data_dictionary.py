ProjectParameters = {
    "RawDataURL" : "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
}

ModelParameters = {
    "RawDataName" : "mnist.npz",

    "Layers" : {
        'numHiddenLayers' : 1,

        0 : {
            'type'    : "Dense",
            'neurons' : 256,
            'activation' : 'relu'
        },
         1 : {
            'type'    : "Dense",
            'neurons' : 120,
            'activation' : 'relu'
        },
        2 : {
            'type'    : "Dense",
            'neurons' : 50,
            'activation' : 'relu'
        }

    },

    "Optimizers" : {
        'name' : 'Adam',
        'LR'   : 0.01
    },

    "FitParameters" : {
        'loss'  : 'SparseCategoricalCrossentropy',
        'epochs': 6 
    }

    
}