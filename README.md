# Machine Learning Model for Sign Language Translator

Machine Learning model for the **Sign Language Recognition** task, that translates sign language into text. The model is integrated at the core of a web application, serving as a communication bridge between the hearing-impaired community and those unfamiliar with sign language. Sign language gestures are processed from video capture and converted into text.

More detailed information about the project can be found in the ```project_synthesis.pdf``` and ```project_presentation.pdf``` files under the ```documentation``` directory.

Stack:
  - ML model: **Python**
  - Frontend: **Node.js**
  - Backend: **Angular**
  - Database: **PostgreSQL**

The module diagram of the web application can be observed below:
![alt text](https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/conceptual_diagram.jpeg)

The ML model can be tested and trained using a desktop application developed with **OpenCV** and **Tkinter**.

### Machine Learning Model Architecture

The model I have created for SLR fits in the Multilayer Perceptron (MLP) category (a feed-forward neural network), because it is a fully connected multi-layer neural network, having one input layer, six hidden layers and one output layer. The model was built using the Sequential class of the Keras API and its architecture is presented below:

<img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/model/sign_classifier/info/sign_classifier_model.png" width="40%" height="40%">

The layers of the neural network are:
1. *Input Layer*: It has the shape equal to (42,), which is the number of x and y Landmark coordinates that describe the hand position. It receives the raw input data and does not process it any further, sending it to the subsequent layer without applying any other transformations;
2. *Hidden Layers*: My model contains two hidden layers, each encompassing a group of Dense, BatchNormalization and Dropout layers.
    - **Dense Layer (128 units, ReLU activation)**: A fully connected layer with 128 neurons, where each node receives input from all the nodes of the previous layer. It applies the ReLU activation function to its outputs, which transforms any negative value to zero, helping introduce non-linearity in the model.
    - **Batch Normalization Layer**: Normalizes the inputs of the batch, so they have a mean of 0 and a standard deviation of 1. This layer was added to increase the training speed of the model and to make it less sensitive to the initial weights of the architecture.
    - **Dropout Layer (0.2)**: Randomly sets a part of the input units to 0 during the training time, which aids to prevent overfitting. In this case, 20% of the inputs are ignored.
    - **Dense Layer (64 units, ReLU activation)**: A fully connected layer with 64 neurons. Its functionality is similar to the one of the first Dense Layer, but it narrows down the number of neurons used, limiting the model’s capacity and helping further prevent overfitting.
    - **Batch Normalization Layer**: Has the same functionality as the previous Batch Normalization Layer.
    - **Dropout Layer (0.2)**: Has the same functionality as the previous Dropout Layer.
3. _Output Layer_: It is the final layer of the neural network, having a number of neurons that is equal to the number of classes (i.e. the number of possible outcomes in which the model can classify input data). It applies the Softmax activation function, which is a suitable choice for multi-class classification problems.

### How to run

1. Create a virtual environment and install dependencies

    Coded with `Python 3.10.8`.

    Install `virtualenv`:
    ```
    pip install virtualenv
    ```

    Create a virtual environment:
    ```
    virtualenv virtualenv
    ```

    Activate the virtual environment:
    ```
    virtualenv\Scripts\activate
    ```
    
    Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
   
    Install the `slt-1.0` project package:
    ```
    pip install -e .
    ```
   
2. Run the main desktop application

    ```
    python main.py
    ```
    The desktop application functions in several modes, which can be activated by pressing the respective keys:
    - **N - normal mode**: In the normal mode, the application simply recognizes gestures performed by the left and right hand.
    - **S - save gesture performed to `sign_dataset.csv`**: By pressing S, a snapshot of the current frame is taken, saving the 21 3D Landmarks of the hand, and new data is added to the sign dataset.
    - **L - create a new sign gesture label**: By pressing L, a new window pops up, allowing the user to create a new sign language gesture label for the dataset. This allows the model to be trained on new gestures in the future.

    <img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/select_label.png" width="30%" height="30%">
    <img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/select_label_dropdown_open.png" width="30%" height="30%">

   - **C - choose sign gesture label to perform**: By pressing C, a new window pops up, allowing the user to choose an existing sign language label from the dataset. Moving forwards, all snapshots of the hand that will be taken will correspond to this selected sign language gesture label.

<img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/create_label.png" width="30%" height="30%">

3. Train the model
  
    After the user created a label for a new sign language gesture and they saved
several snapshots of the said gesture to the dataset, they can easily train the model on
the new data. To do so, inside a terminal with the Virtual Environment virtualenv
activated and located in the main directory of the project, run the following command:

    ```
    python model\sign_classifier\train_model.py
    ```

4. Run the web application

    The web application can be accessed at https://slt-frontend-flkvbjjfeqey.a.run.app. After the user gets registered and then authenticates in their respective account, they will get to the home page of the application:

    <img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/homepage.png" width="70%" height="70%">
   
    By pressing the Chat button, the user will be redirected to the Chat page of the application. In this window, the user can start to perform sign language gestures to the camera, which will be interpreted by the model and then converted to text.

    <img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/hello_chat.png" width="70%" height="70%">
