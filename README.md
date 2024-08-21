<img src="./Images/title.jpg" alt="description" width="100%" height="auto">

# Computer Vision Quality Inspection for Thermo Fisher Scientific
[Thermo Fisher Scientific (TFS) in Hillsboro, Oregon](https://www.thermofisher.com/us/en/home/electron-microscopy/nanoports/hillsboro-nanoport.html) manufactures cutting-edge Scanning Electron Microscopes (SEM). Currently, Liquid Metal Ion Source (LMIS) units are manually inspected for quality, a process that is both costly and prone to human error. This project aims to revolutionize this by implementing an automated Computer Vision system to classify LMIS SEM images accurately, reducing labor costs and eliminating human biases. Using Convolutional Neural Network (CNN) modeling via TensorFlow, the final model achieves 80.5% categorical accuracy while maintaining less than 1% false positive rate.

## Data
LMIS SEM images were downloaded from TFS under Non-Disclosure Agreement (NDA):
- 7,019 grayscale SEM image files
- Images pre-labeled and sorted into 5 class folders
- Due to the NDA: 
   - Source image data is not available to the reader; thus, models are not reproducible.
   - Details containing intellectual property are intentionally omitted.


## Methods
Exploratory Data Analysis (EDA):
- Each image in the data set was reviewed for classification error and moved to the correct classification folder if necessary.
   - Collaborated with a subject matter expert at TFS for classification instructions
   - Noted that some images contain features of multiple non-PASS classification.  In these cases, the most dominant feature was chosen for the classification.
- Class imbalance was noted during EDA as shown below.

Data Preparation:
- Image files were randomly moved into training(70%), validation(15%), and test(15%) folders, with each classification represented within subfolders for each set.
- Data augmentation was used on the images during model training to improve sample variance and avoid overfitting

Modeling:
- CNN was chosen as the modeling architecture due to known high performance for Computer Vision applications.
- A Baseline multi-classification model was established and iterated upon with strategic adjustments to hyperparameters.
- Each model was trained and validated with the respective data sets during each epoch.
- After training, each model was tested with the holdout test set and evaluated for accuracy metrics.
- Due to the high compute cost of training, Google Colab was used to train the models.
- The final CNN architecture and hyperparameters were selected based on the highest multi-classification accuracy along with lowest false-positive rate within the PASS class.
- The final CNN had the probability threshold for the PASS class adjusted to limit false positive rate so as not to exceed 1% (for quality purposes, we do not want to ship failing units)

Implementation:
- Because the manufacturing production network at TFS is behind a firewall and cannot access online modeling tools such as Flask or Streamlit, an executable program was created in python for the purpose of classifying production images and logging the results to a CSV file.
- An interactive PowerBI dashboard was developed and provided for the purpose of monitoring production performance of the executable classifier, which queries data from the CSV logfile.

## Results
Class weights were imposed during model training to account for class imbalance shown below:
<img src="./Images/class_dist.jpg" alt="description" width="100%" height="auto">

Here is the model accuracy and PASS FPR trend by model iteration, with some highlights and lowlights called out:
<img src="./Images/model_trend.jpg" alt="description" width="100%" height="auto">

Best performing CNN architecture is shown below:
<img src="./Images/best_architecture.jpg" alt="description" width="100%" height="auto">

Best performing CNN results are shown below:
<img src="./Images/best_results.jpg" alt="description" width="100%" height="auto">

Here is a gif demonstrating operation of the classification executable, which moves unclassified images into classification folders and logs results to a CSV logfile:
<img src="./Images/classifier.gif" alt="description" width="100%" height="auto">

Here is a gif demonstrating the Power BI interactive production dashboard that pulls from the CSV logfile (fictitious data shown, not real TFS production data):
<img src="./Images/dashboard.gif" alt="description" width="100%" height="auto">


## Conclusions
Best Model: CNN with 4 CONV+POOL layers + MLP with 2 hidden layers. (See Notebook for detailed architecture)
- Best Scores:
   - Results after probability threshold was adjusted to obtain PASS false-positive rate = 0.89%:
    - Test Accuracy = 80.48%
    - PASS False-negative Rate = 18.1%
    - PASS AUC = 99%
    - Split AUC = 98%
    - Etch AUC = 98%
    - Damage AUC = 94%
    - Contamination AUC = 93%
- Since the greatest increase in model performance came about by correcting the image labels, and no further architectural / hyperparameter adjustments could break the 85-87% accuracy barrier, I believe the next move to improve the model is to acquire more and better data.


## Recommendations
1. Initiate a production pilot of provided LMIS Image Classification executable and assess its performance vs. human inspection
   - Recommended pilot evaluation plan:
      - Human review every image for 1 month.  After 1 month:
         - If PASS class false-positive rate <1%, discontinue human evaluation of PASS class images as process of record.
         - Continue long-term human review of all non-PASS images as process of record until model can be improved.
2. Monitor script classification results via provided Power BI dashboard
3. Continue to improve the models with more data as outlined below


## Next Steps
1. Continue to improve the model:
   - Need more data!
      - Subject matter expert labeling of at least 3000 images in each class
      - Train with ‘clean’ samples, i.e., no images that could be more than 1 class
   - Continue attempting transfer learning architectures, i.e., ResNet50
2. If adequate model accuracy can be achieved, implement another production pilot to eliminate human inspection of all images


## For More Information
To see the full data analysis check out the [Jupyter Notebook](./LMIS_Classification_Notebook.ipynb) or review the [Presentation](./LMIS_Classification_Presentation.pdf).

For any additional questions please contact Dale DeFord at:
- daledeford@gmail.com
- https://www.linkedin.com/in/dale-deford-81b54092/

## Repository Structure
```
├── images
├── src
├── LMIS_Classification_Notebook.ipynb
├── LMIS_Classification_Presentation.pdf
└── README.md
```