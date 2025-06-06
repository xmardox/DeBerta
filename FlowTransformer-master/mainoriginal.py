#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import os 

import pandas as pd

from framework.dataset_specification import NamedDatasetSpecifications
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from implementations.classification_heads import *
from implementations.input_encodings import *
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer
from implementations.transformers.named_transformers import *

import wandb
# Autenticación (solo si no has hecho `wandb login` en la terminal antes)
wandb.login(key="cbe075e74cf625c3bd882b7f51894c7aacee0f6f")
# Inicializa tu proyecto en wandb
wandb.init(project="flowtransformer-prueba")


encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    CategoricalFeatureEmbed(EmbedLayerType.Dense, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Lookup, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Projection, 16),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    FlattenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),
    CLSTokenClassificationHead(),
    FeaturewiseEmbedding(project=False),
    FeaturewiseEmbedding(project=True),
]

transformers: List[FunctionalComponent] = [
    BasicTransformer(2, 128, n_heads=2),
    BasicTransformer(2, 128, n_heads=2, is_decoder=True),
    GPTSmallTransformer(),
    BERTSmallTransformer()
]

flow_file_path = r"C:\Users\lapla\Downloads\b3427ed8ad063a09_MOHANAD_A4706\b3427ed8ad063a09_MOHANAD_A4706\data"

datasets = [
    ("CSE_CIC_IDS", os.path.join(flow_file_path, "NF-CSE-CIC-IDS2018-v2.csv"), NamedDatasetSpecifications.unified_flow_format, 0.02, EvaluationDatasetSampling.LastRows),
    #("NSL-KDD", os.path.join(flow_file_path, "NSL-KDD.csv"), NamedDatasetSpecifications.nsl_kdd, 0.05, EvaluationDatasetSampling.RandomRows),
    #("UNSW_NB15", os.path.join(flow_file_path, "NF-UNSW-NB15-v2.csv"), NamedDatasetSpecifications.unified_flow_format, 0.025, EvaluationDatasetSampling.LastRows)
]

pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the transformer
ft = FlowTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[0],
                     sequential_model=transformers[0],
                     classification_head=classification_heads[0],
                     params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
#ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)
ft.load_dataset(dataset_name, dataset_path, dataset_specification,  evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent, n_rows=8000)
# Build the transformer model
m = ft.build_model()
m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'], jit_compile=True)

# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=5, steps_per_epoch=64, early_stopping_patience=5)


for epoch_idx, result in enumerate(train_results):
    train_loss = result[0]
    train_accuracy = result[1]

    # Inicializar métricas de evaluación como None por defecto
    f1 = None
    bal_acc = None


# Mostrar los resultados de evaluación
print("Evaluación Final:\n", eval_results)
