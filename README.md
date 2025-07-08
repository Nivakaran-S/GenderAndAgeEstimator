# Face Recognition with Age and Gender Estimation

## Project Overview
This project implements a real-time face recognition system using a webcam to detect human faces, estimate their age, and predict their gender. Built with Python, OpenCV, and a Wide Residual Network (WideResNet), the system leverages deep learning for accurate face detection and demographic estimation. The project showcases proficiency in computer vision, deep learning, and real-time video processing, making it a robust demonstration of machine learning and software engineering skills.

The system processes live video feeds, detects faces using the Haar Cascade Classifier, and employs a pre-trained WideResNet model to predict age and gender. Additionally, it includes a feature to play specific video clips based on the detected age and gender, enhancing its interactivity and potential for real-world applications such as targeted advertising or user profiling.

## Features

- Real-Time Face Detection: Utilizes OpenCV's Haar Cascade Classifier to detect faces in a live video stream with high accuracy.
- Age and Gender Estimation: Employs a pre-trained WideResNet model to predict the age (0-100 years) and gender (Male/Female) of detected faces.
- Dynamic Video Playback: Plays specific video clips from predefined folders based on the predicted age and gender (e.g., males aged 25-30, females aged 25-30).
- Customizable Network Parameters: Allows users to configure the depth and width of the WideResNet model via command-line arguments.
- Interactive Visualization: Displays detected faces with bounding boxes and overlays predicted age and gender labels on the video feed.
Efficient Face Cropping: Crops and resizes detected faces with customizable margins for optimal model input.

Technologies Used

Python: Core programming language for the project.
OpenCV: For face detection, image processing, and video handling.
Keras/TensorFlow: For loading and running the WideResNet model.
NumPy: For efficient numerical operations and array handling.
Argparse: For parsing command-line arguments to customize model parameters.

Prerequisites
To run this project, ensure you have the following installed:

Python 3.8+
OpenCV (opencv-python)
NumPy
Keras with TensorFlow backend
A webcam for real-time video capture
Pre-trained model weights (weights.18-4.06.hdf5) and Haar Cascade XML file (haarcascade_frontalface_alt.xml)

Installation

Clone the Repository:
git clone https://github.com/your-username/face-recognition-age-gender.git
cd face-recognition-age-gender


Install Dependencies:Create a virtual environment and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install opencv-python numpy tensorflow


Download Pre-trained Models:

Place the haarcascade_frontalface_alt.xml file in the pretrained_models/ directory.
Place the weights.18-4.06.hdf5 file (WideResNet weights) in the pretrained_models/ directory.
Ensure the folder structure is as follows:face-recognition-age-gender/
├── pretrained_models/
│   ├── haarcascade_frontalface_alt.xml
│   ├── weights.18-4.06.hdf5
├── 25-30/
├── 30-35/
├── F25-30/
├── main.py
├── model.py
├── README.md




Optional: Prepare Video Clips:

Add video files to the 25-30/, 30-35/, and F25-30/ folders for dynamic playback based on age and gender predictions.



Usage
Run the main script with optional arguments to customize the WideResNet model:
python main.py --depth 16 --width 8


--depth: Depth of the WideResNet model (default: 16).
--width: Width of the WideResNet model (default: 8).

The program will:

Start the webcam and detect faces in real-time.
Display bounding boxes around detected faces with predicted age and gender labels.
Play a random video clip from the corresponding folder if the predicted age and gender match predefined criteria (e.g., males aged 25-30).
Press ESC to exit the program.

Code Structure

main.py: Entry point for the application, handling argument parsing and initiating the face detection loop.
model.py: Defines the WideResNet model architecture for age and gender prediction.
FaceCV class:
Singleton class for face recognition.
Methods for face detection, cropping, and label drawing.
Integrates the WideResNet model and Haar Cascade Classifier.


pretrained_models/: Stores the Haar Cascade XML and WideResNet weights.
Video folders (25-30, 30-35, F25-30): Contain video clips for playback based on predictions.

Example Output
When running the program, the webcam feed will display detected faces with blue bounding boxes and labels showing the predicted age and gender (e.g., "28, Male"). If the predicted demographics match specific criteria, a video clip from the corresponding folder will play in a separate window.
Future Improvements

Model Fine-Tuning: Train the WideResNet model on a custom dataset for improved accuracy.
Multi-Face Handling: Enhance the system to handle multiple faces more robustly.
GUI Integration: Develop a graphical user interface for better user interaction.
Cloud Deployment: Host the model on a server for remote access and scalability.
Additional Features: Add emotion detection or facial expression analysis.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code follows PEP 8 guidelines and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

The WideResNet model is inspired by the work of [original authors or repository, if applicable].
Haar Cascade Classifier is provided by OpenCV.
Thanks to the open-source community for providing robust libraries like OpenCV and Keras.


Built by [Your Name] to demonstrate expertise in computer vision and deep learning. Connect with me on [LinkedIn/GitHub] for more projects!