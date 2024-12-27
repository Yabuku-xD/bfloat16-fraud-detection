This project is inspired by the paper <b>"A novel approach to increase scalability while training machine learning algorithms using Bfloat 16 in credit card fraud detection" by Bushra Yousuf, Rejwan Bin Sulaiman, and Musarrat Saberin Nipun</b>. <br>It focuses on building and training machine learning models to detect fraudulent transactions using a dataset of financial transaction records. It explores multiple neural network architectures and precision methods, including the ReLU activation function and bfloat16 precision, and analyzes their performance in terms of both accuracy and computational efficiency.

### Key Features
#### Dataset:
The dataset <b>"PS_20174392719_1491204439457_log.csv"</b> contains transaction records with features like type, amount, oldbalanceOrg, newbalanceOrig, and labels for isFraud and isFlaggedFraud.
#### Preprocessing:
- Features are preprocessed using techniques such as scaling and encoding to prepare the data for modeling.
- SMOTE is used to address class imbalance in the dataset.
#### Model Architectures:
- Model 1: Standard neural network with ReLU activation function.
- Model 2: Neural network using bfloat16 precision for performance analysis.
#### Training Comparison:
- ReLU: Achieved model training in ~19 minutes.
- bfloat16: Took ~30 minutes for training due to precision overhead.
#### Performance Metrics:
Accuracy, precision, recall, and F1-score were computed to evaluate model performance.
### Tools and Technologies
Programming Languages: Python

### Libraries:
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, TensorFlow/Keras
- Oversampling: SMOTE from imblearn

### Model Serialization
The trained models are saved for future use:
- Model Architecture: Saved as a JSON file.
- Model Weights: Saved in HDF5 format.

### Observations
- The standard ReLU model performed faster in training compared to the bfloat16 model.
- Hardware optimization is crucial for leveraging bfloat16 precision effectively.
- SMOTE significantly improved model performance on the imbalanced dataset.

### Future Work
- Implement mixed-precision training to optimize bfloat16 performance.
- Experiment with additional neural network architectures and hyperparameter tuning.
- Deploy the trained model as a real-time fraud detection API.
