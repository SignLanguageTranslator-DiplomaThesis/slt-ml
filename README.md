# Machine Learning Model for Sign Language Translator

Machine Learning model for the **Sign Language Recognition** task, that translates sign language into text. The model is integrated at the core of a web application (Node.js/Angular), serving as a communication bridge between the hearing-impaired community and those unfamiliar with sign language. Sign language gestures are processed from video capture and converted into text.

The module diagram of the web application can be observed below:
![alt text](https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/conceptual_diagram.jpeg)
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
    The desktop application functions in several modes, which can be activated
by pressing the respective keys:
    - **N - normal mode**: In the normal mode, the application simply recognizes gestures
performed by the left and right hand.
    - **S - save gesture performed to `sign_dataset.csv`**: By pressing S, a snapshot of the currrent frame is
taken, saving the 21 3D Landmarks of the hand, and new data is added to the sign
dataset.
    - **L - create a new sign gesture label**: By pressing L, a new window pops up, allowing
the user to create a new sign language gesture label for the dataset. This allows
the model to be trained on new gestures in the future.
<img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/select_label.png" width="50%" height="50%">
<img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/select_label_dropdown_open.png" width="50%" height="50%">
   - **C - choose sign gesture label to perform**: By pressing C, a new window pops
up, allowing the user to choose an existing sign language label from the dataset.
Moving forwards, all snapshots of the hand that will be taken will correspond to
this selected sign language gesture label.
<img src="https://github.com/SignLanguageTranslator-DiplomaThesis/slt-ml/blob/main/screenshots/create_label.png" width="50%" height="50%">

3. Train the model
    
    ```
    python model\sign_classifier\train_model.py
    ```