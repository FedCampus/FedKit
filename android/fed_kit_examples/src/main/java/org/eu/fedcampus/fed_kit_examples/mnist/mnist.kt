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

fun sampleSpec() = SampleSpec<FloatArray, FloatArray>(
    { it.toTypedArray() },
    { it.toTypedArray() },
    { Array(it) { FloatArray(1) } },
    ::maxSquaredErrorLoss,
    { samples, logits ->
        samples.zip(logits).forEach { (sample, logit) ->
            Log.d(
                TAG,
                "actual: ${sample.label.contentToString()}, predicted: ${logit.contentToString()}"
            )
        }
        classifierAccuracy(samples, logits)
    },
)

private suspend fun processSet(dataSetDir: String, call: suspend (Int, String) -> Unit) {
    withContext(Dispatchers.IO) {
        File(dataSetDir).useLines {
            it.forEachIndexed { i, l -> launch { call(i, l) } }
        }
    }
}

suspend fun loadData(
    dataDir: String, flowerClient: FlowerClient<FloatArray, FloatArray>, partitionId: Int
) {
    // proecess training set
    Log.i(TAG, "loading pmdata")
    processSet("$dataDir/p${partitionId.toString().padStart(2,'0')}_train.csv") { index, line ->
        addSample(flowerClient, line, true)
    }
    // process test set
    processSet("$dataDir/test.csv") { index, line ->
        addSample(flowerClient, line, false)
    }
}

private fun addSample(
    flowerClient: FlowerClient<FloatArray, FloatArray>, line: String, isTraining: Boolean
) {

    val splits = line.split(",")
    val label = floatArrayOf(splits[splits.size-1].toFloat())
    val featureArray = FloatArray(splits.size-2)
    for (i in featureArray.indices){
        featureArray[i] = splits[i+1].toFloat()
    }
    flowerClient.addSample(featureArray, label, isTraining)
}

private const val TAG = "MNIST Data Loader"
private const val IMAGE_SIZE = 28
private const val LENGTH_ENTRY = IMAGE_SIZE * IMAGE_SIZE
private const val NORMALIZATION = 255f
