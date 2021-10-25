# Examinee-Examiner Network: Weakly Supervised Accurate Coronary Lumen Segmentation using Centerline Constraint

The implementation of Examinee-Examiner Network: Weakly Supervised Accurate Coronary Lumen Segmentation using Centerline Constraint.

## Environment Setup

All the code has been run and tested on  Python 3.7.1, PyTorch 1.6 and numpy 1.16.5, GTX2080Ti GPUs.


## Results presentation

<img src="pic/results.jpg" alt="results" style="zoom:30%;" />



<div align="center"><img src="pic/result1.gif" alt="result1" style="zoom:33%;" /><img src="pic/result2.gif" alt="result2" style="zoom:33%;" /></div>

## Data description and preprocessing

### **Cardiac CCTA Data** (Our private dataset)

- Due to privacy issues, the training data cannot be made public at present, but we supplemented the description and the annotation details of the data to facilitate readers to have an objective understanding of our private data set.
- **「Source and acquisition protocols」** **Cardiac CCTA Data** is multi-center and multi-device data. CCTA images from 132 patients whose stenoses were found in the major epicardial coronary arteries by CTA were selected. Two hospitals in China (Shanghai Jiao Tong University Affiliated Sixth People’s Hospital, Shanghai and Beijing Anzhen Hospital, Capital Medical University, Beijing) provided the imaging data. For the data from the first hospital, a 128-slice multidetector CT (Definition AS, Siemens Medical Solutions, Forchheim, Germany) was employed for scanning.
- **「Detailed information」** According to the data of 132 cases used in the experiment, the age of patients ranged from 38 to 83 years old, with an average age of 62.2 years. Among them, there were 99 males and 33 females. In the data, there are 147 lesions of clinical concern, the average quantitative diameter stenosis rate is 45.2%, and the minimum lumen area is 2.13 $mm^2$. Among them, 60 patients had stenosis with a quantitative diameter stenosis rate of more than 50%

### **ASOCA Data** (The public dataset)

- **ASOCA Data** comes from the 2020 MICCAI Challenge "Automated Segmentation of Coronal Arteries". They provided a training data set of 40 CCTA images with contrast agent showing the coronary arteries, comprising 20 healthy patients and 20 patients with confirmed coronary artery disease. The training data set is used in our paper as the test data to verify the generalization ability of our method on the cross-dataset tasks. Annotations produced by three expert annotators are provided for this data set. Data was collected using retrospective ECG-gated acquisition (GE LightSpeed 64 slice CT Scanner, USA. The time point used for the challenge is late diastole (75\% cardiac cycle)
- Download the ASOCA dataset from the official [website](https://asoca.grand-challenge.org/Home/)

### Data preprocessing

- To adapt to the different data types such as DCM, NRRD, etc. and unify the training process, we changed all the data to the RAW type, and the resolution was re-sampled to 1: 1: 1.
- The original image was $512\times512$ per slice, with 200 to 500 slices per image. The region of interest (ROI) was extracted and these heart ROIs were cropped into small 3D patches of $288\times288\times224$ online before been fed into the network due to the limitation of GPU memory.
- The **「Data」** folder contains two original image from ASOCA **「1.nrrd」****「21.nrrd」** and two RAW image **「1.raw」****「21.raw」**  after our processing. The ROI image after our cutting is provided as a sample, and the size of the image is stored in the **「Npy folder」** as  **「1.npy 」** file**「21.npy」** file.

<div align="center"><img src="pic/28nrrd.png" alt="28nrrd" style="zoom:50%;" /><img src="pic/28nrrd2d.png" alt="28nrrd2d" style="zoom:30%;" /></div>

- Taking ASOCA data as an example, we first add 1024 uniformly, so that CT values can be converted into gray values for unified processing. Then we use the pre-trained model for predicting the heart region to predict the heart chamber and cut out the ROI region. Here we provide the processed samples **「1.raw」** and **「21.raw」**. Meanwhile, the provided images are also normalized.

## Network

![EE-Net](pic/EE-Net.jpg)

### Framework

- The basic network framework of our Examinee and Examiner network adopts the 3D U-Net architecture, in which the Examinee contains four down-sampling and the Examiner network contains three down-sampling.
- For details, please refer to the **「E1_Net」** and **「E2_Net」**  section under the **「code」** folder.

## Prediction

- Please refer to the **「EEE_Learning」**  under the **「code」** folder.

- **「Npy」** stores the size of the cropped images. **「Weights_m」** provides our trained parameters for direct use. 

- Specific comments are in the code.

  ```python
  model1_name = 'Evaluation1_66_1_new.dat'
  test_image_dir = 'Txt/test_challenge_image.txt' test_label_dir = 'Txt/test_challenge_label.txt'
  
  save_dir = 'Results/R_challenge'    
  checkpoint_dir2 = 'Weights_m'  
     net1.load_state_dict(torch.load(os.path.join(checkpoint_dir2, model1_name))) 
      
  predict_all(net1, save_path=save_dir, shape=shape, img_path=test_image_dir, num_classes=1)
  ```

## Training and learning strategy

- Please refer to the **「EEE_Learning」**  under the **「code」** folder.

-  **「DataLoader_txt」**is used for processing the input data.

  ```python
  model1.train()
  model2.train()
  for batch_idx in range(loader1.__len__()):
      input1, target11, target12 = loader1.__iter__().__next__() # full-supervised data
      input2, target22 = loader2.__iter__().__next__() # weakly-supervised data
  
      if torch.cuda.is_available():
         input1 = input1.cuda()
         input2 = input2.cuda()
         target11 = target11.cuda()
         target12 = target12.cuda()
         target22 = target22.cuda()
  
         optimizer1.zero_grad()
         optimizer2.zero_grad()
         model1.zero_grad()
         model2.zero_grad()
  ```

  

  ```python
  # stage1
  output12 = model2(target11)
  loss12 = criterion2(target12, output12)
  
  losses12.update(loss12.data, target12.size(0))
  loss1 = loss12
  
  loss1.backward()
  optimizer2.step()
  
  # stage2
  output21 = model1(input1)
  output22 = model2(output21)
  
  loss21 = criterion1(target11, output21)
  loss22 = criterion2(target12, output22)
  
  loss2 = loss21 + loss22
  
  losses21.update(loss21.data, target11.size(0))
  losses22.update(loss22.data, target12.size(0))
  
  loss2.backward()
  optimizer1.step()
  
  # stage3
  output31 = model1(input2)
  output32 = model2(output31)
  
  loss32 = criterion2(target22, output32)
  
  losses32.update(loss32.data, target22.size(0))
  
  loss3 = loss32
  
  loss3.backward()
  optimizer1.step()
  ```

  

