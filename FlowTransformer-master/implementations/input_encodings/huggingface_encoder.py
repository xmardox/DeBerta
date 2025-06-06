from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)

# Usamos los imports de TensorFlow para evitar conflictos con Keras puro
Input = tf.keras.layers.Input
Lambda = tf.keras.layers.Lambda

class HuggingFaceTextEncoder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.required_input_format = "textual"
        self.model_name = model_name  # para usar luego al cargar modelo

    def encode(self, inputs):
        return self.tokenizer(
            inputs,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

    def __call__(self, inputs):
        encodings = self.encode(inputs)
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    def build(self, window_size, model_input_spec):
        # No es necesario hacer nada acá por ahora
        pass

    def apply(self, m_inputs, prefix=""):
        input_ids = None
        attention_mask = None

        # Identificar los tensores por nombre
        for inp in m_inputs:
            if "input_ids" in inp.name:
                input_ids = inp
            elif "attention_mask" in inp.name:
                attention_mask = inp

        if input_ids is None or attention_mask is None:
            raise ValueError("Input tensors 'input_ids' and 'attention_mask' not found in model inputs")

        # Cargar el modelo DeBERTa
        model = TFAutoModel.from_pretrained(self.model_name)

        # Capa Lambda que ejecuta el modelo y devuelve el CLS token (primer vector de la secuencia)
         # Capa Lambda que ejecuta el modelo y devuelve el CLS token (primer vector de la secuencia)
        def hf_model_layer(x):
            ids, mask = x
            outputs = model(input_ids=ids, attention_mask=mask)
            return outputs.last_hidden_state[:, 0, :]


        # Aplicar la capa Lambda con output_shape explícito
        output = Lambda(
            hf_model_layer,
            name=f"{prefix}deberta",
            output_shape=(768,)  # solo el vector [CLS]
        )([input_ids, attention_mask])

        return output