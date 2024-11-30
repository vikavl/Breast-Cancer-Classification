# Breast-Cancer-Classification
Breast cancer is a leading cause of death among women worldwide. Early and precise diagnosis is critical for effective treatment and improving patient outcomes. Potentialy improve diagnostic tools and assist healthcare professionals in decision-making.

---
## **1. Dataset Overview**
The **[CBIS-DDSM (Curated Breast Imaging Subset of the Digital Database for Screening Mammography)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data)** is a large-scale dataset used for **breast cancer diagnosis**, particularly focusing on **mammography images**. It is available on **Kaggle**, and it provides images along with annotations that are essential for training and evaluating machine learning models for tasks such as **tumor classification** (benign vs. malignant) and **lesion detection** in breast tissue.

The CBIS-DDSM dataset is a curated subset of the **Digital Database for Screening Mammography (DDSM)**. It was specifically curated for **breast cancer detection** using mammography images and contains both **mammogram images** and corresponding **annotations**.

### **i. Key Features**:
- **Image Type**: The dataset includes **digitized mammogram images** which are used to detect and classify breast cancer.
- **Focus**: The dataset is focused on **cancerous and non-cancerous lesions** in mammography images.
- **Annotations**: Along with images, the dataset contains **annotations** marking regions of interest (ROI), such as tumors or abnormalities in the breast tissue.

### **ii. Dataset Contents**
The CBIS-DDSM dataset consists of the following key components:

**A. Image Data**:
- The dataset contains **mammogram images** (X-ray scans of the breast tissue) that are typically used for the **screening** and **early detection** of breast cancer.
- The images are typically in **DICOM format**, which is a standard format used in medical imaging.
- Each image is usually annotated with a **region of interest (ROI)**, identifying abnormal areas in the breast tissue.
  
**B. Labels and Annotations**:
- **Labels**: Each image is labeled as **benign** or **malignant**, indicating the classification of the detected lesions.
- **Annotations**: For each mammogram image, there are **bounding box annotations** to highlight the tumor region or other abnormalities.
  - **Lesion Types**: Each lesion is classified into different types such as **mass**, **calcification**, or **architectural distortion**.
  
**C. Metadata**:
- The dataset provides additional metadata for each case, such as the **patient’s age**, **image size**, **image resolution**, and other contextual details that may assist in the analysis.


### **iii. Dataset Labels and Classifications**
Each image in the CBIS-DDSM dataset can be classified as either:

- **Benign**: The lesion is non-cancerous.
- **Malignant**: The lesion is cancerous and needs attention.

The lesions within the images may also be classified into specific categories:

- **Mass**: A dense region in the breast that could be benign or malignant.
- **Calcifications**: Small deposits of calcium that can be indicative of benign or malignant changes.
- **Architectural Distortion**: Changes in the tissue pattern, often associated with malignancy.

### **iv. Dataset Size**

The dataset contains **hundreds of mammogram images**, and it is large enough to support deep learning experiments. For example, some datasets have over **1,000 annotated mammogram images**. Each image typically contains multiple **regions of interest (ROIs)**.

---
