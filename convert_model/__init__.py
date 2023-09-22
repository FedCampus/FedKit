import tensorflow as tf

keras = tf.keras


def red(displayed) -> str:
    return f"\033[91m{displayed}\033[0m"
