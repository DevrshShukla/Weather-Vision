🛰 Satellite Image Classification App
An interactive Streamlit web application for satellite image classification. This tool allows you to explore datasets, train deep learning models, visualize results, and make predictions on satellite images.

Built for machine learning practitioners, researchers, and students to easily experiment with image classification tasks using a clean and modern UI.

🚀 Features
✅ Data Overview

Load and preview satellite image datasets.

View dataset statistics and class distributions.

Download dataset CSV for offline use.

✅ Image Gallery

![image](https://github.com/user-attachments/assets/8b7e3623-0240-4dd5-82c1-10b405089d8f)

Model Training 
![image](https://github.com/user-attachments/assets/c7a51d84-187d-4e3e-acd4-7e2b5f158411)

Results & Metrics
![image](https://github.com/user-attachments/assets/421849a3-328a-427f-be82-105712cabfc3)




Browse sample images by class.

View image statistics (dimensions, formats).

✅ Model Training

Train deep learning models with configurable parameters (epochs, batch size, learning rate).

Choose from multiple architectures: CNN, ResNet50, VGG16, MobileNet.

Simulated progress bar for real-time feedback.

✅ Results & Metrics

View training accuracy, validation accuracy, and loss curves.

Inspect confusion matrix and detailed classification report.

✅ Predictions

Upload single images for classification.

Batch prediction support (ZIP uploads).

View predicted class, confidence scores, and probability distributions.

✅ Dark Theme

Modern dark UI for better usability and aesthetics.

📸 Screenshots


Results & Metrics	Predictions

📦 Installation
🔗 Clone Repository
bash
Copy
Edit
git clone https://github.com/<your-username>/satellite-image-classification.git
cd satellite-image-classification
📥 Install Dependencies
We recommend creating a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
▶ Run the App
Start the Streamlit app:

bash
Copy
Edit
streamlit run app.py
The app will open in your browser at http://localhost:8501.

📂 Project Structure
bash
Copy
Edit
satellite-image-classification/
│
├── app.py                  # Streamlit app
├── requirements.txt        # Python dependencies
├── assets/                 # Screenshots & static assets
├── dataset/                # Sample dataset (or path to your dataset)
└── README.md               # Project documentation
📊 Dataset
Default Dataset Path: /content/dataset/Satellite Image data/

Expected structure:

kotlin
Copy
Edit
Satellite Image data/
├── cloudy/
├── desert/
├── green_area/
└── water/
Each folder contains images for the respective class.

You can replace this with your own dataset following the same folder structure.

⚙ Configuration
Modify labels_config in app.py to change class names.

Adjust training parameters from the Model Training page in the app.

🛠 Tech Stack
Streamlit – Interactive UI

Plotly – Visualizations

Seaborn – Statistical graphics

scikit-learn – Metrics & evaluation

Pillow – Image processing

NumPy & Pandas – Data handling

🏗 Future Improvements
✅ Add real model training (currently uses simulated metrics).

✅ Support TensorFlow/Keras for actual training and inference.

✅ Implement batch prediction for ZIP file uploads.

✅ Add support for more satellite image datasets (e.g., EuroSAT, UC Merced).

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Fork the project

Create your feature branch (git checkout -b feature/awesome-feature)

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/awesome-feature)

Open a Pull Request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Developed by Devrsh Shukla
📧 dshuklagls@gmail.com
