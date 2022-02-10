from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --------------------------
class TensorflowLiteClassificationModel:

    def __init__(self, model_path, labels, image_size=224):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size = image_size

    def run_from_filepath(self, input_image):
        input_data_type = self._input_details[0]["dtype"]

        image = np.array(input_image.resize(
            (self.image_size, self.image_size)),
                         dtype=input_data_type)
        """

        image = np.array(np.resize(input_image,
            (self.image_size, self.image_size, 3)),
                         dtype=input_data_type)
        """
        if input_data_type == np.float32:
            image = image / 255.

        if image.shape == (1, 224, 224):
            image = np.stack(image * 3, axis=0)

        if image.shape == (224, 224, 3):
            image = np.expand_dims(image, axis=0)

        return self.run(image)

    def run(self, image):
        """
        args:
          image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """

        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(
            self._output_details[0]["index"])
        probabilities = list(tflite_interpreter_output[0])

        # create list of ["label", probability], ordered descending probability
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        return sorted(label_to_probabilities, key=lambda element: element[1])
        #return label_to_probabilities

@app.get("/")
def index():
    return {"greeting": "Working API"}

@app.post("/predict")
def predict(img_file: UploadFile = File(...)):
    model_path = "model_edge.tflite"
    #labels = [1, 3, 2, 0]
    labels = [2, 0, 4]
    model = TensorflowLiteClassificationModel(model_path, labels)
    input_image = Image.open(img_file.file)
    label_probability = model.run_from_filepath((input_image))
    return label_probability
