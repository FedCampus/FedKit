import tensorflow as tf

keras = tf.keras


def red(string: str) -> str:
    return f"\033[91m{string}\033[0m"
