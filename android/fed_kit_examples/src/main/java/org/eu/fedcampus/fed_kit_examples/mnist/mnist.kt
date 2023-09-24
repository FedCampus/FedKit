package org.eu.fedcampus.fed_kit_examples.mnist

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.eu.fedcampus.fed_kit_examples.cifar10.Float3DArray
import org.eu.fedcampus.fed_kit_train.FlowerClient
import org.eu.fedcampus.fed_kit_train.SampleSpec
import org.eu.fedcampus.fed_kit_train.helpers.classifierAccuracy
import org.eu.fedcampus.fed_kit_train.helpers.maxSquaredErrorLoss
import java.io.File

fun sampleSpec() = SampleSpec<Float3DArray, FloatArray>(
    { it.toTypedArray() },
    { it.toTypedArray() },
    { Array(it) { FloatArray(N_CLASSES) } },
    ::maxSquaredErrorLoss,
    ::classifierAccuracy,
)

private suspend fun processSet(dataSetDir: String, call: suspend (Int, String) -> Unit) {
    withContext(Dispatchers.IO) {
        File(dataSetDir).useLines {
            it.forEachIndexed { i, l -> launch { call(i, l) } }
        }
    }
}

suspend fun loadData(
    dataDir: String, flowerClient: FlowerClient<Float3DArray, FloatArray>, partitionId: Int
) {
    processSet("$dataDir/MNIST_train.csv") { index, line ->
        if (index / 6000 + 1 != partitionId) {
            return@processSet
        }
        if (index % 1000 == 999) {
            Log.i(TAG, "Prepared $index training data points.")
        }
        addSample(flowerClient, line, true)
    }
    processSet("$dataDir/MNIST_test.csv") { index, line ->
        if (index % 1000 == 999) {
            Log.i(TAG, "Prepared $index test data points.")
        }
        addSample(flowerClient, line, false)
    }
}

private fun addSample(
    flowerClient: FlowerClient<Float3DArray, FloatArray>, line: String, isTraining: Boolean
) {
    val splits = line.split(",")
    val feature = Array(IMAGE_SIZE) { Array(IMAGE_SIZE) { FloatArray(1) } }
    val label = FloatArray(N_CLASSES)
    for (i in 0 until LENGTH_ENTRY) {
        feature[i / IMAGE_SIZE][i % IMAGE_SIZE][0] = splits[i].toFloat() / NORMALIZATION
    }
    label[splits.last().toInt()] = 1f
    flowerClient.addSample(feature, label, isTraining)
}

private const val TAG = "MNIST Data Loader"
private const val IMAGE_SIZE = 28
private const val N_CLASSES = 10
private const val LENGTH_ENTRY = IMAGE_SIZE * IMAGE_SIZE
private const val NORMALIZATION = 255f
