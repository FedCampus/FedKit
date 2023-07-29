package org.eu.fedcampus.train.db

import android.content.Context
import java.io.File

// Always change together with Python `train.models.TFLiteModel`.
data class TFLiteModel(
    val id: Long,
    val name: String,
    val file_path: String,
    val layers_sizes: IntArray,
) {
    @Throws
    fun getModelDir(context: Context): File {
        return context.getExternalFilesDir("models/$name/")!!
    }
}
