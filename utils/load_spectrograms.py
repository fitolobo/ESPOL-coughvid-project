import pickle
import gc
import numpy as np


def load_large_pickle(pickle_path, chunk_size=1000):
    """
    Carga un archivo pickle grande de manera eficiente.

    Args:
        pickle_path (str): Ruta al archivo pickle
        chunk_size (int): Número de elementos a procesar por vez

    Returns:
        dict: Diccionario con los datos del espectrograma
    """
    try:
        # Abrir el archivo en modo binario
        with open(pickle_path, "rb") as file:
            # Cargar los datos
            data = pickle.load(file)

            # Forzar la recolección de basura para liberar memoria
            gc.collect()

            return data

    except Exception as e:
        print(f"Error al cargar el archivo pickle: {str(e)}")
        return None


def inspect_pickle_structure(pickle_path):
    """
    Inspecciona la estructura de un archivo pickle sin cargarlo completamente.

    Args:
        pickle_path (str): Ruta al archivo pickle
    """
    try:
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)

            # Obtener información básica
            print(f"Tipo de datos: {type(data)}")

            if isinstance(data, dict):
                print(f"Número de elementos: {len(data)}")

                # Mostrar un ejemplo de la estructura
                first_key = list(data.keys())[0]
                print(f"\nEstructura del primer elemento (clave: {first_key}):")
                print_nested_structure(data[first_key])

            gc.collect()

    except Exception as e:
        print(f"Error al inspeccionar el archivo pickle: {str(e)}")


def print_nested_structure(obj, level=0):
    """
    Imprime la estructura anidada de un objeto.
    """
    prefix = "  " * level

    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{prefix}{key}:")
            print_nested_structure(value, level + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}Lista/Tupla de longitud {len(obj)}")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}Array de forma {obj.shape}, tipo {obj.dtype}")
    else:
        print(f"{prefix}Valor de tipo {type(obj)}")
