import numpy as np
import nilearn.image
#import nilearn.plotting
import time

DATA = '/data/gallussb/2_mlproject/DATA/'
# original mri
TRAINING_DATA = DATA + 'set_train/train_%d.nii'
TEST_DATA = DATA + 'set_test/test_%d.nii'
# segmented mri
#TRAINING_DATA = DATA + 'train_segmented/mri/mwp1train_%d.nii'
#TEST_DATA = DATA + 'test_segmented/mri/mwp1test_%d.nii'
TARGETS = 'targets.csv'

SLICES_LIMIT = 15 # this influences the runtime
TRAINING_SAMPLES = 278 #278 #ranging from 1 to 278
TEST_SAMPLES = 138 #138 ranging from 1 to 138

# Disregard black areas around the brain
CUT_0 = (16,160)
CUT_1 = (20,188)
CUT_2 = (11,155)

def training_file(n):
    return TRAINING_DATA % n
def test_file(n):
    return TEST_DATA % n

#backdrop = nilearn.image.load_img(training_file(1)) # For visualisation purposes only

def standardize_data(X,Z):
    complete_data = np.concatenate((X, Z), axis=0) #this doesn't copy the array
    means = np.mean(complete_data, axis=0)
    stds = np.std(complete_data, axis=0)
    #
    def standardize_feature(feature, mean, std):
        if std == 0: #all pixels at location i,j have the same value
            return feature - mean
        else:
            return (feature - mean)/std
    def standardize_array(A):
        for i in range(means.shape[0]):
            for j in range(means.shape[1]):
                A[:,i,j] = standardize_feature(A[:,i,j], means[i,j], stds[i,j])
        return A

    X = standardize_array(X)
    Z = standardize_array(Z)
    return (X,Z)

def one_hot_encode_targets(y):
    # modify targets for classification
    y_one_hot_encoded = np.zeros((len(y),8), dtype = np.float32)
    for i, y_i in enumerate(y):
        binary_sum = 4 * y_i[0] + 2 * y_i[1] + y_i[2]
        y_one_hot_encoded[i, int(binary_sum)] = 1.0
    return y_one_hot_encoded

def load_data(max_training_samples):
    # training data
    X = np.squeeze(np.stack([nilearn.image.load_img(training_file(n)).get_data()
                    for n in range(1,TRAINING_SAMPLES+1)], axis = 0))[:max_training_samples] # delete the fifth dimension, the "time"-dimension doesn't exist
    # print(X.shape)
    model_axes = np.array(X.shape[1:]) / 2 # that's the middle slice of each dimension

    # THREE DIFFERENT AXES
    # X_augmented = np.stack([X[n,model_axes[0] + m,:,:]
    # X_augmented = np.stack([X[n,:,model_axes[1] + m,:]
    X_augmented = np.stack([X[n,CUT_0[0]:CUT_0[1],CUT_1[0]:CUT_1[1],model_axes[2] + m]
    	for n in range(X.shape[0]) # We don't use TRAINING_SAMPLES, because there can be less samples due to max_training_samples
    		for m in range(-SLICES_LIMIT,SLICES_LIMIT+1)] , axis = 0).astype(np.float32) # slice along the z-axis (up-down). We take only the SLICES_LIMIT * 2 central slices of the dimension
    #print("max {} min {}".format(np.max(X_augmented[0]), np.min(X_augmented[0])))

    # test data
    Z = np.squeeze(np.stack([nilearn.image.load_img(test_file(n)).get_data()
                    for n in range(1,TEST_SAMPLES+1)], axis = 0)) # delete the fifth dimension, the "time"-dimension doesn't exist

    # test data shouldn't be augmented. We just take the middle slice
    # Z_augmented = np.stack([Z[n,:,:,model_axes[2]] for n in range(TEST_SAMPLES)], axis = 0).astype(np.float32)

    # THREE DIFFERENT AXES
    # Z_augmented = np.stack([Z[n,model_axes[0] + m,:,:]
    # Z_augmented = np.stack([Z[n,:,model_axes[1] + m,:]
    Z_augmented = np.stack([Z[n,CUT_0[0]:CUT_0[1],CUT_1[0]:CUT_1[1],model_axes[2] + m]
    	for n in range(TEST_SAMPLES)
    		for m in range(-SLICES_LIMIT,SLICES_LIMIT+1)] , axis = 0).astype(np.float32) # slice along the z-axis (up-down). We take only the SLICES_LIMIT * 2 central slices of the dimension
    # print(Z_augmented.shape)

    X_augmented, Z_augmented = standardize_data(X_augmented, Z_augmented)


    # targets for training data
    # binary value for each of the three classes
    y = np.genfromtxt(TARGETS, delimiter=',')[:TRAINING_SAMPLES][:max_training_samples]
    print(y.shape)
    y = one_hot_encode_targets(y)
    print(y.shape)

    y_augmented = np.array([y_n for y_n in y for m in range(-SLICES_LIMIT,SLICES_LIMIT+1)], dtype = np.float32) # has to be float32 for Theano

    # # triple the data with [0,1] target
    # X_oversampling = []
    # y_oversampling = []
    # for i,y_a in enumerate(y_augmented):
    #     if (y_a == np.array([0.0, 1.0])).all():
    #         X_oversampling.append(X_augmented[i])
    #         X_oversampling.append(X_augmented[i])
    #         y_oversampling.append(y_a)
    #         y_oversampling.append(y_a)
    #
    # if(X_oversampling): # only concatenate if there is oversampled samples
    #     X_augmented = np.concatenate((X_augmented, np.array(X_oversampling)))
    #     y_augmented = np.concatenate((y_augmented, np.array(y_oversampling)))

    #print(y_augmented)
    print("data was loaded")
    return {'X': X_augmented, 'y': y_augmented, 'Z': Z_augmented, 'SLICES_LIMIT': SLICES_LIMIT}
