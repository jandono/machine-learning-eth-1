# Goal: produce average brains of
#
#   old/young
#   male/female
#   healthy/sick
#
#   and combinations...

DATA = '../data'


import nilearn.image
import nilearn.plotting
import numpy as np

#Load some images
def load_all_as_imgs(typ):
    all = []
    for n in range(278):
        print('loading ' + str(n + 1))
        path = '%s/set_%s/%s_%d.nii' % (DATA, typ, typ, n + 1)
        img = nilearn.image.load_img(path)
        all.append(img)
    return all

X = np.array(load_all_as_imgs('train'))
y = np.genfromtxt(DATA + '/targets.csv', delimiter=',')

labels_names = ['female', 'young', 'healthy']

# In which ways can the labels be combined
combinations = [
    (0,), (1,), (2,),
    (0, 1), (1, 2), (2, 0),
    (0, 1, 2) ]

# In which ways the groups of labels can be inverted
invert = {
    1: [(0,), (1,)],
    2: [(0,0), (0,1), (1,0), (1,1)],
    3: [(0,0,0),
        (0,0,1), (0,1,0), (1,0,0),
        (0,1,1), (1,0,1), (1,1,0),
        (1,1,1)]
}


# For each combination of labels
for combination in combinations:
    # and for each potential inversion of those labels
    inversions = invert[len(combination)]
    for inversion in inversions:
        final_data = X
        final_labels = y
        #output_label_names = labels_names[:] # copy
        output_label_names = []

        # Filter each label accordingly
        for i, (label_ind, take_false) in enumerate(zip(combination, inversion)):
            if take_false: # Take data with this label set at "false"
                which = final_labels[:, label_ind] == 0
                output_label_names.append("N" + labels_names[label_ind])
            else:
                which = final_labels[:, label_ind] == 1
                output_label_names.append(labels_names[label_ind])


            final_data = final_data[which]
            final_labels = final_labels[which]

        # Actually do the mean and save it to a file
        if len(final_data) > 0:
            mean_img = nilearn.image.mean_img(final_data)
            title = '_'.join(output_label_names)
            fname = 'means/' + title
            print('Saving', title)

            mean_img.to_filename(fname + '.nii')

            nilearn.plotting.plot_anat(mean_img,
                title=title,
                output_file=fname + '.png',
                cut_coords=[0,0,0])


