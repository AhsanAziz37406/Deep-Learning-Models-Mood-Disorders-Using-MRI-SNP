******Deep Learning Models Mood Disorders Using MRI SNP*******

1. MATLAB (the project is 100% MATLAB files in the repo). Confirm and install a modern release (R2019b / R2020a / R2021a or later recommended). The repo language is MATLAB. 
GitHub

2. Recommended toolboxes:

⦁	Deep Learning Toolbox
⦁	Image Processing Toolbox
⦁	Statistics and Machine Learning Toolbox
⦁	(Optional) Parallel Computing Toolbox for speedups

3. Data required:

⦁	MRI data as DICOM or NIfTI (scripts include loadDicomFile.m and loadNiftiFile.m / Merge_nii_G.m).
⦁	Genetic/SNP data in CSV format (csv_class.m and genetic_data_classification.mat). 

****If you don’t have the toolboxes, some code will error****


Mental and mood disorders, such as Major Depressive Disorder (MDD), are influenced by both **neurobiological** and **genetic** factors. This project implements a deep learning framework that combines **Magnetic Resonance Imaging (MRI)** and **Single Nucleotide Polymorphism (SNP)** data to identify early biomarkers and improve diagnostic accuracy.

The pipeline includes:
- MRI preprocessing and slice/frame extraction  
- Deep CNN feature extraction  
- Genetic (SNP) data preprocessing and feature normalization  
- Fusion of MRI and genetic features  
- Training and evaluation of classifiers for both **child** and **adult** datasets  
- Hyperparameter optimization using metaheuristic algorithms (Genetic Algorithm and Ant Colony Optimization)



#### ****Core Workflow****

The following stages describe the complete end-to-end implementation:

**1. Data Acquisition**

The project uses:
- **MRI Scans**: DICOM or NIfTI images from subjects diagnosed with or without mood disorders.  
- **SNP/Genetic Data**: CSV files containing gene-level SNP data corresponding to the same subjects.

Each subject’s MRI and genetic data are linked through a **subject ID** key to ensure correct multimodal fusion.

**2. MRI Preprocessing and Frame Extraction**

Raw MRI volumes are first standardized and converted into a usable format:

- DICOM and NIfTI files are loaded using:
- `loadDicomFile.m` for `.dcm`
- `loadNiftiFile.m` or `Merge_nii_G.m` for `.nii`
- Non-brain or irrelevant slices are removed.
- 3D MRI volumes are converted into 2D slices for deep learning using:
- `frame_Extraction.m` or `child_normal_frame_extraction.m`

Each extracted slice is resized and normalized to a consistent image dimension suitable for CNN models (e.g., 224×224).


**3. Feature Extraction from MRI**

The imaging pipeline supports multiple deep feature extraction strategies:

These layers model complex inter-region relationships within MRI slices, enabling advanced classification.


**4. Genetic (SNP) Data Processing**

The SNP data are handled using:

Processing steps include:
- Data cleaning and handling of missing values  
- Feature scaling and normalization  
- Encoding genetic variations for each subject  
- Preparing the feature matrix for integration with MRI features


**5. Multimodal Fusion of MRI and Genetic Features**

After both modalities are processed:
1. MRI embeddings are extracted from the trained CNN model.
2. SNP feature vectors are aligned by subject ID.
3. Features are concatenated (or passed through a fusion layer) to form a combined multimodal representation.

This fusion is implemented in:
- `fused.m`
- `fused_child_adult.m`

The fusion improves model robustness by integrating neuroimaging biomarkers with genetic signatures.


**6. Classification of Mood Disorders**

Two main categories are modeled:

1. **Child Classification**
   - `Child_Normal_Abnormal_Classification.m`
   - `Child_MDD_Classification.m`
   These scripts distinguish between normal and abnormal (or MDD vs. healthy) cases using CNN architectures.

2. **Adult Classification**
   - `both_classification.m`
   - `adult_both_classification_External_validation_for_fused.m`
   Similar flow but trained on adult subject datasets and externally validated using separate test data.

Each classification model outputs metrics such as:
- Accuracy  
- Sensitivity / Specificity  
- Confusion matrix  
- ROC curves (if enabled in the script)


**7. Hyperparameter Optimization**

To maximize model performance, optimization algorithms are used:
- **Ant Colony Optimization (ACO)** (`jACO.m`)
- Custom `fitnessFunction.m` to evaluate model accuracy and generalization.

These algorithms automatically tune learning rates, dropout rates, hidden layer sizes, and other key hyperparameters to improve accuracy and prevent overfitting.


**8. Model Evaluation**

The final trained models (e.g., `trainedClassifier_combined_child_adult_v3.mat`) are validated on unseen datasets.

Validation steps include:
- Confusion matrix visualization
- Calculation of key performance metrics

This stage confirms the generalization capability of the proposed model across independent samples.


