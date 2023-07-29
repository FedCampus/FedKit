package org.eu.fedcampus.benchmark

import android.content.Context
import android.util.Log
import androidx.work.WorkerParameters
import org.eu.fedcampus.train.background.BaseTrainWorker
import org.eu.fedcampus.train.examples.cifar10.DATA_TYPE
import org.eu.fedcampus.train.examples.cifar10.Float3DArray
import org.eu.fedcampus.train.examples.cifar10.loadData
import org.eu.fedcampus.train.examples.cifar10.sampleSpec

class BenchmarkCifar10Worker(context: Context, params: WorkerParameters) :
    BaseTrainWorker<Float3DArray, FloatArray>(
        context,
        params,
        R.drawable.ic_launcher_foreground,
        sampleSpec(),
        DATA_TYPE,
        ::loadData,
        ::logTrain
    ) {
    companion object {
        const val TAG = "BenchmarkCifar10Worker"
    }
}

fun logTrain(msg: String) {
    Log.i(BenchmarkCifar10Worker.TAG, msg)
}
