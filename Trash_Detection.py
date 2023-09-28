#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns; sns.set()

dataset_path = r'\Users\Anant Roop Mathur\Downloads\Waste_Detection_Dataset\data'
anns_file_path = dataset_path + '/' + 'annotations.json'

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []
super_cat_names = []
super_cat_ids = {}
super_cat_last_name = ''
nr_super_cats = 0
for cat_it in categories:
    cat_names.append(cat_it['name'])
    super_cat_name = cat_it['supercategory']
    # Adding new supercat
    if super_cat_name != super_cat_last_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        super_cat_last_name = super_cat_name
        nr_super_cats += 1

print('Number of super categories:', nr_super_cats)
print('Number of categories:', nr_cats)
print('Number of annotations:', nr_annotations)
print('Number of images:', nr_images)


# In[2]:


anns[0]


# In[3]:


# Counting the annotations 
cat_histogram = np.zeros(nr_cats,dtype=int)
for ann in anns:   # Ann denotes annotation
    cat_histogram[ann['category_id']] += 1

# Initialize the matplotlib figure
fig, ax = plt.subplots(figsize=(5,15))

# Convert the annotations into a DataFrame
df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations',0,False) # Sorting in a descending order of count as for a better visualization

# Plotting the histogram comparing the number of annotations
plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df,
            label="Total", color="b", orient='h')


# # Converting into larger categories
# * What we have done is converted different small categories and grouped them to form a large category or super category

# In[4]:


cat_ids_2_supercat_ids = {}
for cat in categories:
    cat_ids_2_supercat_ids[cat['id']] = super_cat_ids[cat['supercategory']]

# Count annotations (Similar to that of smaller categories)
super_cat_histogram = np.zeros(nr_super_cats,dtype=int)
for ann in anns:
    cat_id = ann['category_id']
    super_cat_histogram[cat_ids_2_supercat_ids[cat_id]] +=1
    
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(5,10))

# Convert to DataFrame
d ={'Super categories': super_cat_names, 'Number of annotations': super_cat_histogram}
df = pd.DataFrame(d)
df = df.sort_values('Number of annotations', 0, False)

# sns.set_color_codes("pastel")
# sns.set(style="whitegrid")
plot_1 = sns.barplot(x="Number of annotations", y="Super categories", data=df,
            label="Total", color="b", orient='h')


# ### Analysis of the background scene
# * What we are doing now is trying to analyse the background of each scene and trying to define and categorize the different scenes/images on the basis of the background.
# * This is helpful as it will be easier for us to visualize the scene as we will be given context. We can say certain type of waste usually occurs in certain areas.
# Ex:- Near Water(Seas, Oceans etc.), we usually find plastic bottles. We can visualize that if the background is water than there is a good chance that plastic bottle might be the waste detected. 
# * Although, this doesn't mean that there can't be any other type of waste present in that particular area, it's just for our better understanding.

# In[5]:


# We are going to draw a pie chart comparing different types of backgrounds
scene_cats = dataset['scene_categories']
scene_name = []
for scene_cat in scene_cats:
    scene_name.append(scene_cat['name'])

nr_scenes = len(scene_cats)
scene_cat_histogram = np.zeros(nr_scenes,dtype=int)

for scene_ann in dataset['scene_annotations']:    
    scene_ann_ids = scene_ann['background_ids']
    for scene_ann_id in scene_ann_ids:
        if scene_ann_id<len(scene_cats):
            scene_cat_histogram[scene_ann_id]+=1

# Convert to DataFrame
df = pd.DataFrame({'scene_cats': scene_cats, 'nr_annotations': scene_cat_histogram})
 
# Plot
colors = ['darkblue','darkred','darkgreen', 'gold', 'red','lightgreen','lightskyblue']
plt.pie(scene_cat_histogram, labels=scene_name, colors = colors,
      shadow=False, startangle=-120)
 
plt.axis('equal')
plt.show()


# # Viewing the annotation in an image
# In this code, we specify an image file path and loads it along with its corresponding annotations from a COCO dataset. The code handles potential image orientation adjustments based on Exif metadata and displays the image. It then overlays the annotations on the image using random colors, showing both the annotated regions as filled polygons and bounding boxes around objects. The final result is an image with visually highlighted annotations.

# In[6]:


from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab


image_filepath = 'batch_1/000008.jpg'

pylab.rcParams['figure.figsize'] = (28,28)

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
# Loading the  dataset as a coco object
coco = COCO(anns_file_path)

# Find image id
img_id = -1
for img in imgs:
    if img['file_name'] == image_filepath:
        img_id = img['id']
        break

# Show image and corresponding annotations
if img_id == -1:
    print('Incorrect file name')
else:

    # Load image
    print(image_filepath)
    I = Image.open(dataset_path + '/' + image_filepath)
     # Load and process image metadata
    if I._getexif():
        exif = dict(I._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                I = I.rotate(180,expand=True)
            if exif[orientation] == 6:
                I = I.rotate(270,expand=True)
            if exif[orientation] == 8:
                I = I.rotate(90,expand=True)

    # Show image
    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(I)
# Load mask ids
    annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
    anns_sel = coco.loadAnns(annIds)

    # Show annotations
    for ann in anns_sel:
        color = colorsys.hsv_to_rgb(np.random.random(),1,1)
        for seg in ann['segmentation']:
            poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
            p = PatchCollection([poly], facecolor=color, edgecolors=color,linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        [x, y, w, h] = ann['bbox']
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '--')
        ax.add_patch(rect)
    plt.show()


# # Visualizing Selected Category in a COCO Dataset
# We load and display a specified number of images containing a chosen category or super-category from a COCO dataset. We first obtains the Exif orientation tag code, loads the COCO dataset, and retrieves image IDs based on the provided category name. Then, we randomly selects a subset of images, processes them to handle orientation if needed, and display them one by one with the associated annotations overlaid. Annotations are represented as filled polygons and bounding boxes, with each image having a distinct color scheme, providing a visual representation of objects belonging to the chosen category within the dataset.

# In[7]:


from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab

nr_img_2_display = 10
category_name = 'Cigarette'
pylab.rcParams['figure.figsize'] = (14, 14)

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

coco = COCO(anns_file_path)
imgIds = []
catIds = coco.getCatIds(catNms=[category_name])
if catIds:
    imgIds = coco.getImgIds(catIds=catIds)
else:
    catIds = coco.getCatIds(supNms=[category_name])
    for catId in catIds:
        imgIds += (coco.getImgIds(catIds=catId))
    imgIds = list(set(imgIds))

nr_images_found = len(imgIds)
print('Number of images found: ', nr_images_found)

random.shuffle(imgIds)
imgs = coco.loadImgs(imgIds[0:min(nr_img_2_display, nr_images_found)])
for img in imgs:
    image_path = dataset_path + '/' + img['file_name']
    I = Image.open(image_path)
    if I._getexif():
        exif = dict(I._getexif().items())
        if orientation in exif:
            if exif[orientation] == 3:
                I = I.rotate(180, expand=True)
            if exif[orientation] == 6:
                I = I.rotate(270, expand=True)
            if exif[orientation] == 8:
                I = I.rotate(90, expand=True)

    fig, ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(I)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns_sel = coco.loadAnns(annIds)
    for ann in anns_sel:
        color = colorsys.hsv_to_rgb(random.random(), 1, 1)
        for seg in ann['segmentation']:
            poly = Polygon(np.array(seg).reshape((int(len(seg) / 2), 2)))
            p = PatchCollection([poly], facecolor=color, edgecolors=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        [x, y, w, h] = ann['bbox']
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', alpha=0.7, linestyle='--')
        ax.add_patch(rect)

    plt.show()


# In[8]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen_train = ImageDataGenerator(rescale = 1/255, shear_range = 0.2, zoom_range = 0.2, 
                               brightness_range = (0.1, 0.5), horizontal_flip=True)

train_data = gen_train.flow_from_directory(r'\Users\Anant Roop Mathur\Downloads\Waste_Detection_Dataset\data',
                                           target_size = (224, 224), batch_size = 32, class_mode="categorical")


# In[9]:


# let's create a model
# here i'm going to use VGG16 model's parameter to solve this problem

from tensorflow.keras.applications.vgg16 import VGG16

# here i'm going to take input shape, weights and bias from imagenet and include top False means
# i want to add input, flatten and output layer by my self

vgg16 = VGG16(input_shape = (224, 224, 3), weights = "imagenet", include_top = False)


# In[10]:


for layer in vgg16.layers:
  layer.trainable = False


# In[11]:


from tensorflow.keras import layers

x = layers.Flatten()(vgg16.output)


# In[12]:


prediction = layers.Dense(units = 9, activation="softmax")(x)

# creating a model object

model = tf.keras.models.Model(inputs = vgg16.input, outputs=prediction)
model.summary()


# In[13]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics =["accuracy"])


# In[14]:


from tensorflow.keras.preprocessing import image
output_class = ["Plastic bag & wrapper","Cigarette","Unlabeled litter","Bottle","Bottle cap","Can","Other plastic","Straw","Paper","Broken glass","Syrofoam piece","Pop tab","Lid","Plastic container","Aluminium Foil","Plastic utensils","Rope & Strings","Paper bag","Scrap metal","Food waste","Shoe","Squeezable tube","Blister pack","Glass jar","Plastic glooves","Battery"]
def waste_prediction(new_image):
  test_image = image.load_img(new_image, target_size = (224,224))
  plt.axis("off")
  plt.imshow(test_image)
  plt.show()
 
  test_image = image.img_to_array(test_image) 
  test_image = np.expand_dims(test_image, axis=0) / 255

  predicted_array = model.predict(test_image)
  predicted_value = output_class[np.argmax(predicted_array)]

  print("Your waste material is ", predicted_value)


# In[15]:


waste_prediction(r"C:\Users\Anant Roop Mathur\Downloads\Waste_Detection_Dataset\data\batch_1/000008.jpg")


# In[ ]:




