import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.keras import TqdmCallback

def gen_ds(df, target, seq_length:int, seed:int=None, shuffle:bool=False):

    timestep_ds = tf.keras.utils.timeseries_dataset_from_array(
        df.to_numpy(),
        targets=df[target][seq_length:],
        sequence_length=seq_length,
        seed=seed,
        shuffle=shuffle,
    )
    return timestep_ds

def build_ds(df_normalized, target, seq_length, period: str, shuffle: bool, batch_size) -> tuple[tf.data.Dataset]:
    train_ds_array = []
    val_ds_array = []

    for timestep in period["treino"]:
            train_ds_array.append(
                tf.keras.utils.timeseries_dataset_from_array(
                    df_normalized[timestep[0] : timestep[1]].to_numpy(),
                    targets=df_normalized[timestep[0] : timestep[1]][target][seq_length :],
                    sequence_length=seq_length,
                    shuffle=shuffle,
                    batch_size=batch_size,
                )
            )
    for timestep in period["validacao"]:
            val_ds_array.append(
                tf.keras.utils.timeseries_dataset_from_array(
                    df_normalized[timestep[0] : timestep[1]].to_numpy(),
                    targets=df_normalized[timestep[0] : timestep[1]][target][seq_length :],
                    sequence_length=seq_length,
                    shuffle=shuffle,
                    batch_size=batch_size,
                )
            )

    train_ds = train_ds_array[0]
    val_ds = val_ds_array[0]

    for dataset in train_ds_array[1:]:
        train_ds = train_ds.concatenate(dataset)

    for dataset in val_ds_array[1:]:
        val_ds = val_ds.concatenate(dataset)

    return train_ds, val_ds


def normalize_df(data: pd.DataFrame, variables: list) -> pd.DataFrame:
    return (data[variables] - data[variables].min()) / (data[variables].max() - data[variables].min())

def denormalize_df(target_df: pd.DataFrame, original_df: pd.DataFrame, variables: list) -> pd.DataFrame:
    if variables[0] == "real":
        return (target_df * (original_df["pressao_na_choke"].max() - original_df["pressao_na_choke"].min()) + original_df["pressao_na_choke"].min())
    
    return (target_df * (original_df[variables].max() - original_df[variables].min()) + original_df[variables].min())

def create_sequences(df, target, sequence_length):
    X, y = [], []
    
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i + sequence_length].values)
        y.append(df.iloc[i + sequence_length][target])
    
    return np.array(X), np.array(y)

def multi_step_prediction(model: tf.keras.Model, normalized_df, features, target, seq_length, periodo: list) -> pd.DataFrame:
        df_periodo = normalized_df.query(
            "index >= @periodo[0] and index <= @periodo[1]"
        )

        train_X, train_y = create_sequences(df_periodo, target, seq_length)

        len_analise = train_y.shape[0]
        predictions = np.zeros_like(train_y[:len_analise])

        input_atual = train_X[0:1]
        prediction = model.predict(input_atual, verbose=0)
        predictions[0] = prediction

        for i in tqdm(range(1, len_analise)):
            novo_input = np.zeros_like(input_atual)
            novo_input[0, 0:-1] = input_atual[0, 1:]
            novo_input[0, -1, :] = train_X[i : i + 1, -1, :]

            i_variavel_predicao =features.index(target)
            novo_input[0, -1, i_variavel_predicao] = predictions[i - 1]

            input_atual = novo_input
            prediction = model.predict(input_atual, verbose=0)
            predictions[i] = prediction

        indices_datahora =normalized_df.query(
            "index >= @periodo[0] and index <= @periodo[1]"
        ).index[seq_length : ]

        data = {'real': train_y, 'predito': predictions}

        return pd.DataFrame(data, index=indices_datahora)