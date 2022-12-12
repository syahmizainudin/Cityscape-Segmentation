# %%
import tensorflow as tf
from tensorflow import keras
from keras.layers import RandomFlip, Input, Conv2DTranspose, Concatenate, Layer
from keras.utils import plot_model
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, glob, datetime

# %% 0. Dataset meta information
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


# %% 1. Data loading
DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

train_path = glob.glob(os.path.join(DATASET_PATH, 'train', '*.jpg'))
val_path = glob.glob(os.path.join(DATASET_PATH, 'val', '*.jpg'))

# %% 2. Data cleaning and visualization
IMAGE_SIZE = (128,128)
ID2COLOR = { label.id : np.asarray(label.color) for label in labels }

def find_closest_labels_vectorized(mask, mapping):     
    closest_distance = np.full([mask.shape[0], mask.shape[1]], 10000) 
    closest_category = np.full([mask.shape[0], mask.shape[1]], None)   

    for id, color in mapping.items():
        dist = np.sqrt(np.linalg.norm(mask - color.reshape([1,1,-1]), axis=-1))
        is_closer = closest_distance > dist
        closest_distance = np.where(is_closer, dist, closest_distance)
        closest_category = np.where(is_closer, id, closest_category)
    
    return closest_category

train_features = [cv2.resize(cv2.cvtColor(cv2.imread(image)[:, :256], cv2.COLOR_BGR2RGB), IMAGE_SIZE) for image in train_path]
train_targets = [cv2.resize(cv2.cvtColor(cv2.imread(mask)[:, 256:], cv2.COLOR_BGR2RGB), IMAGE_SIZE) for mask in train_path]
train_targets_enc = [find_closest_labels_vectorized(mask, ID2COLOR) for mask in train_targets]

val_features = [cv2.resize(cv2.cvtColor(cv2.imread(image)[:, :256], cv2.COLOR_BGR2RGB), IMAGE_SIZE) for image in val_path]
val_targets = [cv2.resize(cv2.cvtColor(cv2.imread(mask)[:, 256:], cv2.COLOR_BGR2RGB), IMAGE_SIZE) for mask in val_path]
val_targets_enc = [find_closest_labels_vectorized(mask, ID2COLOR) for mask in val_targets]

np_train_features = np.array(train_features)
np_train_targets = np.stack(train_targets_enc).astype('float32')

np_val_features = np.array(val_features)
np_val_targets = np.stack(val_targets_enc).astype('float32')

# %%
# Visualize an image and its mask
def display(image_list):
    plt.figure(figsize=(10,10))
    title = ['Original image', 'Encoded mask', 'Predicted mask']

    for i in range(len(image_list)):
        if i != 0:
            if type(image_list[i]).__name__ == 'EagerTensor':
                enc = image_list[i].numpy()
            else:
                enc = image_list[i]

            mask = np.zeros([enc.shape[0], enc.shape[1], 3])

            for row in range(enc.shape[0]):
                for col in range(enc.shape[1]):
                    mask[row, col, :] = ID2COLOR[enc[row, col].item()]
                    mask = mask.astype('uint8')
            image = mask
        else:
            image = image_list[i]

        plt.subplot(1, len(image_list), i+1)
        plt.title(title[i])
        plt.axis(False)
        plt.imshow(image)
    plt.show()

display([np_val_features[213], np_val_targets[213]])
display([np_train_features[213], np_train_targets[213]])

# %% 3. Data preprocessing
# Expand the dimension of targets
np_train_targets = np.expand_dims(np_train_targets, -1)
np_val_targets = np.expand_dims(np_val_targets, -1)

# Normalize features
np_train_features = np_train_features / 255.0
np_val_features = np_val_features / 255.0

# Convert numpy array to tensor
train_features_tensor = tf.data.Dataset.from_tensor_slices(np_train_features)
train_targets_tensor = tf.data.Dataset.from_tensor_slices(np_train_targets)
val_features_tensor = tf.data.Dataset.from_tensor_slices(np_val_features)
val_targets_tensor = tf.data.Dataset.from_tensor_slices(np_val_targets)

# Combine train label and features into zip dataset
train = tf.data.Dataset.zip((train_features_tensor, train_targets_tensor))
val = tf.data.Dataset.zip((val_features_tensor, val_targets_tensor))

# %%
# Data augmentation layer
SEED = 12345
TRAIN_SIZE = len(train)
BATCH_SIZE = 64
BUFFER_SIZE = 3000
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE

class Augment(Layer):
    def __init__(self, seed=SEED):
        super().__init__()
        self.augment_image = RandomFlip('horizontal', seed=seed)
        self.augment_mask = RandomFlip('horizontal', seed=seed)

    def __call__(self, image, mask):
        image = self.augment_image(image)
        mask = self.augment_mask(mask)
        return image, mask

# Convert dataset to prefetch dataset
train_batches = (
    train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_batches = val.batch(BATCH_SIZE)

# Display image from train batches
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

# %% 4. U-Net Model development
INPUT_SHAPE = list(IMAGE_SIZE) + [3,]

# Define pretrained model
base_model = keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)
plot_model(base_model, show_shapes=True, show_layer_names=True)

# Activation layer outputs from base model for concatenation
layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu', 
    'block_16_project'
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Feature extractor
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Upsampling path
up_stack = [
    pix2pix.upsample(512,3),
    pix2pix.upsample(256,3),
    pix2pix.upsample(128,3),
    pix2pix.upsample(64,3)
]

# Build U-Net
OUTPUT_CLASSES = len(labels)

# Downsampling layer
inputs = Input(shape=INPUT_SHAPE)
skips = down_stack(inputs)
x = skips[-1]
skips = reversed(skips[:-1])

# Upsampling layers
for up,skip in zip(up_stack, skips):
    x = up(x)
    concat = Concatenate()
    x = concat([x,skip])

# Transpose convolution layer as output
last = Conv2DTranspose(filters=OUTPUT_CLASSES, kernel_size=3, strides=2, padding='same')
outputs = last(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)

# Model compiling
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics='acc')

# %%
# Create function for displaying prediction
def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(tf.expand_dims(sample_image, axis=0)))])

# %%
# Create a callback function with the show_predictions function
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample predictions after epoch {}\n'.format(epoch+1))

LOG_DIR = os.path.join(os.getcwd(), "logs", datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# %%
# Model training
EPOCHS = 50
VAL_SUBSPLITS = 2
VALIDATION_STEPS = len(val) // BATCH_SIZE // VAL_SUBSPLITS

model_history = model.fit(
    train_batches, 
    validation_data=val_batches, 
    validation_steps=VALIDATION_STEPS, 
    epochs=EPOCHS, 
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[DisplayCallback(), tb]
)

# %% 5. Model evaluation
show_predictions(val_batches, num=5)

# %%
