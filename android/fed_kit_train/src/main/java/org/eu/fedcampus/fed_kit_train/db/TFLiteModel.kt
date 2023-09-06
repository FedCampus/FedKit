package org.eu.fedcampus.fed_kit_train.db

import android.content.Context
import java.io.File

// Always change together with Python `train.models.TFLiteModel`.
data class TFLiteModel(
    val id: Long,
    val name: String,
    val tflite_path: String,
    val tflite_layers: IntArray,
) {
    @Throws
    fun getModelDir(context: Context): File {
        return context.getExternalFilesDir("models/$name/")!!
    }
}
