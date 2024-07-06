#!/usr/bin/env python
# coding: utf-8

# In[1]:




import numpy as np 
import pandas as pd


# In[2]:


import os
for dirname, _, filenames in os.walk("C:\\Users\\91701\\Downloads\\archive (4)"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from pathlib import Path
import os
import cv2
import glob
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm


# In[5]:


pip install torchvision


# In[6]:


get_ipython().system('pip install albumentations > /dev/null 2>&1')


# In[ ]:





# In[7]:


pip install albumentations


# In[8]:


pip install pretrainedmodels


# In[9]:


get_ipython().system('pip install albumentations')
get_ipython().system('pip install pretrainedmodels')


# In[10]:


get_ipython().system('which python')


# In[11]:


conda activate your_environment_name


# In[12]:


import albumentations


# In[13]:


train_df = pd.read_csv("C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\train.csv")
test_df = pd.read_csv("C:\\Users\\91701\\Downloads\\archive (4)\\test_vc2kHdQ.csv")
submit = pd.read_csv("C:\\Users\\91701\\Downloads\\archive (4)\\sample_submission_yxjOnvz.csv")


# In[14]:


train_df.shape, test_df.shape


# In[15]:


train_df.groupby('emergency_or_not').count()


# In[16]:


sns.countplot(x='emergency_or_not' , data=train_df)


# In[18]:


data_folder = Path("C:\\Users\\91701\\Downloads\\archive (4)")
data_path = "C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\images"

path = os.path.join(data_path , "*jpg")


# In[19]:


data_path


# In[20]:


files = glob.glob(path)
data=[]
for file in files:
    image = cv2.imread(file)
    data.append(image)


# In[21]:


train_images = data[:1646]
test_images= data[1646:]


# In[22]:


print(train_images[0].shape), print(train_images[100].shape)


# In[23]:


def get_images_class(cat):
    list_of_images = []
    fetch = train_df.loc[train_df['emergency_or_not']== cat][:3].reset_index()
    for i in range(0,len(fetch['image_names'])):
        list_of_images.append(fetch['image_names'][i])
    return list_of_images 


# In[24]:


get_images_class(0)


# In[25]:


get_images_class(1)


# In[26]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
fig = plt.figure(figsize=(20,15))
for i, image_name in enumerate(get_images_class(0)):
    plt.subplot(1,3 ,i+1)
    img=mpimg.imread("C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\images\\"+image_name)
    imgplot = plt.imshow(img)
    plt.xlabel(str("Non-Emergency Vehicle") + " (Index:" +str(i+1)+")" )
plt.show()


# In[27]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
fig = plt.figure(figsize=(20,15))
for i, image_name in enumerate(get_images_class(1)):
    plt.subplot(1,3 ,i+1)
    img=mpimg.imread("C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\images\\"+image_name)
    imgplot = plt.imshow(img)
    plt.xlabel(str("Emergency Vehicle") + " (Index:" +str(i)+")" )
plt.show()


# In[28]:


class EmergencyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['image_names'], row['emergency_or_not']
        img_fname = self.root_dir + str(img_id)
#         + ".jpg"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, img_label


# In[29]:


TRAIN_CSV = "C:\\Users\\91701\\Downloads\\archive (4)\\train_SOaYf6m\\train.csv"
transform = transforms.Compose([transforms.ToTensor()])
dataset = EmergencyDataset(TRAIN_CSV, data_path, transform=transform)


# In[30]:


torch.manual_seed(10)

val_pct = 0.2
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size


# In[31]:


import torch
from torch.utils.data import random_split


# In[32]:


train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# In[33]:


batch_size = 32


# In[34]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# In[35]:


dataset = EmergencyDataset(TRAIN_CSV, data_path, transform=transform)


# In[36]:


import torch
from torch.utils.data import DataLoader


# In[37]:


train_loader  = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

validation_loader = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)


# In[38]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[39]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.4)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*4*4,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.sig = nn.Sigmoid()
        
        
    def forward(self, x):

        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batchnorm2(self.pool(x)))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))

        x = x.view(x.size(0), -1)


        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


# In[40]:


model = Net() # On CPU
#model = Net().to(device)  # On GPU
print(model)


# In[41]:


criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[42]:


def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


# In[43]:


pip install --upgrade torch torchvision


# In[44]:


pip install --upgrade numpy Pillow


# In[45]:


from torch.utils.data import SubsetRandomSampler


# In[46]:


train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
validation_loader = DataLoader(val_ds, batch_size=batch_size*2, num_workers=0, pin_memory=True)


# In[47]:


from torch.utils.data import SubsetRandomSampler

# Assuming train_size and val_size are defined
train_sampler = SubsetRandomSampler(range(train_size))
val_sampler = SubsetRandomSampler(range(train_size, train_size + val_size))

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
validation_loader = DataLoader(dataset, batch_size=batch_size*2, sampler=val_sampler, num_workers=0, pin_memory=True)


# In[ ]:





# In[48]:


import os

file_path = "C:\\Users\\91701\\Downloads\\archive (4)"
if os.path.exists(file_path):
    print(f'The file {file_path} exists.')
else:
    print(f'The file {file_path} does not exist.')


# In[49]:


from PIL import Image
import os

class YourDatasetClass:
    # ... (other methods)

    def __getitem__(self, idx):
        img_id = self.image_list[idx]
        img_fname = os.path.join(self.root_dir, str(img_id) + ".jpg")

        # Handle missing file
        if not os.path.exists(img_fname):
            print(f"Warning: File {img_fname} not found. Skipping.")
            return None

        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


# In[50]:


from PIL import Image
import os

class YourDatasetClass:
    # ... (other methods)

    def __getitem__(self, idx):
        img_id = self.image_list[idx]
        img_fname = os.path.join(self.root_dir, str(img_id) + ".jpg")

        # Check if the file exists
        if not os.path.exists(img_fname):
            print(f"Warning: File {img_fname} not found. Skipping.")
            return None

        # Open the image only if the file exists
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


# In[51]:


n_epochs = 20
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    # scheduler.step(epoch)
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_loader):
        
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
train_acc.append(100 * correct / total)
train_loss.append(running_loss/total_step)
print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
batch_loss = 0
total_t=0
correct_t=0
with torch.no_grad():
model.eval()
for data_t, target_t in (validation_loader):
outputs_t = model(data_t)
loss_t = criterion(outputs_t, target_t)
batch_loss += loss_t.item()
_,pred_t = torch.max(outputs_t, dim=1)
correct_t += torch.sum(pred_t==target_t).item()
total_t += target_t.size(0)
val_acc.append(100 * correct_t / total_t)
val_loss.append(batch_loss/len(validation_loader))
network_learned = batch_loss < valid_loss_min
print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        # Saving the best weight 
if network_learned:
    
    
    
    valid_loss_min = batch_loss
    torch.save(model.state_dict(), 'model_classification.pt')
print('Detected network improvement, saving current model')
model.train()


# In[52]:


n_epochs = 20
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    # scheduler.step(epoch)
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_loader):
        #data_, target_ = data_.to(device), target_.to(device)# on GPU
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        model.eval()
        for data_t, target_t in (validation_loader):
            #data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss/len(validation_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        # Saving the best weight 
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'model_classification.pt')
            print('Detected network improvement, saving current model')
    model.train()


# In[53]:


fig = plt.figure(figsize=(20,10))
plt.title("Train - Validation Loss")
plt.plot( train_loss, label='train')
plt.plot( val_loss, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')


# In[54]:


fig = plt.figure(figsize=(20,10))
plt.title("Train - Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')


# In[56]:


def img_display(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


# In[57]:


dataiter = iter(validation_loader)
images, labels = dataiter.next()
vehicle_types = {0: 'Non-Emergency-Vehicle', 1: 'Emergency-Vehicle'}
# Viewing data examples used for training
fig, axis = plt.subplots(3, 5, figsize=(20, 15))
with torch.no_grad():
    model.eval()
    for ax, image, label in zip(axis.flat,images, labels):
        ax.imshow(img_display(image)) # add image
        image_tensor = image.unsqueeze_(0)
        output_ = model(image_tensor)
        output_ = output_.argmax()
        k = output_.item()==label.item()
        ax.set_title(str(vehicle_types[label.item()])+":" +str(k)) # add label


# In[ ]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('C:\\Users\\91701\\Downloads\\sign_mnist_train.csv')
test=pd.read_csv('C:\\Users\\91701\\Downloads\\sign_mnist_train.csv')
labels=train['label'].values
unique_val=np.array(labels)
np.unique(unique_val)
plt.figure(figsize=(18,2))
sns.countplot(x=labels)
train.drop('label',axis=1,inplace=True)
images=train.values
images=np.array([np.reshape(i,(28,28))for i in images])
images=np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer
label_binrizer=LabelBinarizer()
labels=label_binrizer.fit_transform(labels)
labels
 index=2
print(labels[index])
plt.imshow(images[index].reshape(28,28))

import cv2
import numpy as np
for i in range(0,10):
    rand=np.random.randint(0, len(images))
    input_im=images[rand]
    
    sample=input_im.reshape(28,28).astype(np.uint8)
    sample=cv2.resize(sample,None,fx=10,fy=10,interpolation=cv2.INTER_CUBIC)
    cv2.imshow("sample image",sample)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()   
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.3,random_state=101)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
batch_size=128
num_classes=24
epochs=10   
x_train=x_train/255
x_test=x_test/255
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
plt.imshow(x_train[0].reshape(28,28))
plt.imshow(x_train[0].reshape(28,28))
from tensorflow.keras.layers import Conv2D ,MaxPooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam

model =Sequential()
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer=Adam(),
             metrics=['accuracy'])
print(model.summary())
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size)
model.save("sign_mnist_cnn_50_Epochs.keras")
print("Model saved")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accurachy')
plt.legend(['train','test'])

plt.show()
test_labels=test['label']
test.drop('label',axis=1,inplace=True)

test_images=test.values
test_images=np.array([np.reshape(i,(28,28)) for  i in test_images])
test_images=np.array([i.flatten() for i in test_images])

test_labels=label_binrizer.fit_transform(test_labels)

test_images=test_images.reshape(test_images.shape[0],28,28,1)

test_images.shape
y_pred=model.predict(test_images)
from sklearn.metrics import accuracy_score

accuracy_score(test_labels,y_pred.round())
def getLetter(result):
    classLabels={0:'A',
                 1:'B',
                 2:'C',
                 3:'D',
                 4:'E',
                 5:'F',
                 6:'G',
                 7:'H',
                 8:'I',
                 9:'J',
                 10:'K',
                 11:'L',
                 12:'M',
                 13:'N',
                 14:'O',
                 15:'P',
                 16:'Q',
                 17:'R',
                 18:'S',
                 19:'T',
                 20:'U',
                 21:'V',
                 22:'W',
                 23:'X'}
    try:
        res=int(result)
        return classLabels[res]
    except:
        return "Error"
    
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    else:
        pass
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    roi = frame[100:400, 320:620]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    cv2.imshow('roi scaled and gray', roi)
    
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)
    
    roi = roi.reshape(1, 28, 28, 1)
    
    # Using the 'predict' method
    prediction = model.predict(roi)
    predicted_class = np.argmax(prediction)
    result = str(predicted_class)
    
    cv2.putText(copy, getLetter(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
    
import cv2
import numpy as np
import math


vid = cv2.VideoCapture(0)

while True:
    flag, imgFlip = vid.read()
    img = cv2.flip(imgFlip,cv2.COLOR_BGR2GRAY)

    
    cv2.rectangle(img, (100,100), (300,300), (0,255,0), 0)
    imgCrop = img[100:300, 100:300]


    imgBlur = cv2.GaussianBlur(imgCrop, (3,3), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

    lower = np.array([2,0,0])
    upper = np.array([20,255,255])
    mask = cv2.inRange(imgHSV, lower, upper)

   
    kernel = np.ones((5,5))

   
    dilation = cv2.dilate(mask,kernel, iterations=1)
    erosion = cv2.erode(dilation,kernel, iterations=1)

    filtered_img = cv2.GaussianBlur(erosion, (3,3), 0)
    ret, imgBin = cv2.threshold(filtered_img, 127, 255, 0)


    contours, hierarchy = cv2.findContours(imgBin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key = lambda x: cv2.contourArea(x))

       
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imgCrop, (x,y), (x+w,y+h), (0,0,255), 0)

     
        con_hull = cv2.convexHull(contour)


       
        con_hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, con_hull)
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

           
            if angle<=90:
                count_defects+=1
                cv2.circle(imgCrop, far, 2, [0,0,255], -1)

            cv2.line(imgCrop, start, end, [0,255,0], 2)

        if count_defects == 0:
            cv2.putText(img, "ONE", (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
        elif count_defects == 1:
            cv2.putText(img, "TWO", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 2:
            cv2.putText(img, "THREE", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 3:
            cv2.putText(img, "FOUR", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        elif count_defects == 4:
            cv2.putText(img, "FIVE", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        else:
            pass

    except:
        pass

   

    cv2.imshow("Gesture", img)
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()    

