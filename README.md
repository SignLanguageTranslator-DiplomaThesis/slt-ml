# Machine Learning Model for Sign Language Translator

Model that recognizes and classifies various sign language gestures.

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
    The application has several modes, each activated by pressing the respective key:
    - **N** - normal mode
    - **S** - save gesture performed to `sign_dataset.csv`
    - **L** - create a new sign gesture label
    - **C** - choose sign gesture label to perform

3. Train the model
    
    ```
    python model\sign_classifier\train_model.py
    ```