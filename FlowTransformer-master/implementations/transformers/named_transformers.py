#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from framework.base_sequential import BaseSequential
from implementations.transformers.basic.decoder_block import TransformerDecoderBlock
from implementations.transformers.basic.encoder_block import TransformerEncoderBlock


class GPTSmallTransformer(BaseSequential):

    @property
    def name(self) -> str:
        return "GPT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = True

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerDecoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate)(m_x)

        return m_x


class BERTSmallTransformer(BaseSequential):
    
    @property
    def name(self) -> str:
        return "BERT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 12
        self.internal_size = 768
        self.n_heads = 12
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = False

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            #BERT comparison 
            m_x = TransformerEncoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, prefix=f"block_{layer_i}_")(m_x,training=True)
           
           #m_x = TransformerEncoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, use_conv=self.use_conv, prefix=f"{prefix}block_{layer_i}_")(m_x, training=True)

        return m_x
    
# implementations/transformers/deberta_transformer.py

from framework.base_sequential import BaseSequential
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf


class DeBERTaTransformer(BaseSequential):
    @property
    def name(self) -> str:
        return "DeBERTa Base"

    @property
    def parameters(self) -> dict:
        return {
            "model_name": self.model_name
        }

    def __init__(self, model_name="microsoft/deberta-base"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.deberta_model = TFAutoModel.from_pretrained(model_name)
        self.output_size = self.deberta_model.config.hidden_size

    def apply(self, X, prefix: str = None):
  
        if isinstance(X, dict) and "input_ids" in X and "attention_mask" in X:
            from tensorflow.keras.layers import Lambda

            def deberta_lambda(inputs):
                input_ids, attention_mask = inputs
                # Devuelve directamente el last_hidden_state
                return self.deberta_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            # Ejecuta el modelo en tiempo de ejecución con Lambda
            hidden_states = Lambda(
                deberta_lambda,
                name=f"{prefix}deberta_output",
                output_shape=(128, 768)
            )([X["input_ids"], X["attention_mask"]])

            # Extrae el [CLS] token (posición 0) de cada secuencia
            from tensorflow.keras.layers import Lambda
            cls_output = Lambda(lambda x: x[:, 0], name=f"{prefix}cls_token_output")(hidden_states)


            return cls_output

        else:
            raise ValueError("El input X debe contener input_ids y attention_mask")















    # def apply(self, X, prefix: str = None):
    #     """
    #     X: un batch de texto o ya tokenizado. 
    #     Este método debe adaptarse según el encoding que uses en FlowTransformer.
    #     """
    #     # X debe venir como input_ids y attention_mask para DeBERTa
    #     # Si X ya es dict con 'input_ids' y 'attention_mask', los pasamos directo
    #     if isinstance(X, dict) and "input_ids" in X and "attention_mask" in X:
    #         #outputs = self.deberta_model(input_ids=X["input_ids"], attention_mask=X["attention_mask"])
    #         from tensorflow.keras.layers import Lambda

    #         def deberta_lambda(inputs):
    #             input_ids, attention_mask = inputs
    #             return self.deberta_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    #         # Agregar capa Lambda que llama al modelo DeBERTa dentro del grafo Keras
    #         outputs = Lambda(deberta_lambda, name=f"{prefix}deberta_output", output_shape=(128, 768))(
    #             [X["input_ids"], X["attention_mask"]]
    #         )

    #     else:
    #         raise ValueError("El input X debe contener input_ids y attention_mask")

    #     # Salida del CLS token (primera posición)
    #     cls_output = outputs.last_hidden_state[:, 0, :]
    #     return cls_output
