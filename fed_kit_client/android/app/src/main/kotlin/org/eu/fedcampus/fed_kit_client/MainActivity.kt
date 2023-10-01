package org.eu.fedcampus.fed_kit_client

import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.EventChannel.EventSink
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import org.eu.fedcampus.fed_kit_examples.cifar10.Float3DArray
import org.eu.fedcampus.fed_kit_examples.mnist.loadData
import org.eu.fedcampus.fed_kit_examples.mnist.sampleSpec
import org.eu.fedcampus.fed_kit_train.FlowerClient
import org.eu.fedcampus.fed_kit_train.helpers.loadMappedFile
import java.io.File
import java.nio.ByteBuffer


class MainActivity : FlutterActivity() {
    val scope = MainScope()
    lateinit var flowerClient: FlowerClient<Float3DArray, FloatArray>
    var events: EventSink? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        MethodChannel(messenger, "fed_kit_client_mnist_ml_client").setMethodCallHandler(::handle)
        EventChannel(messenger, "fed_kit_client_mnist_ml_client_log").setStreamHandler(object :
            EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, eventSink: EventSink?) {
                if (eventSink === null) {
                    Log.e(TAG, "onListen: eventSink is null.")
                } else {
                    events = eventSink
                    Log.d(TAG, "onListen: initialized events.")
                }
            }

            override fun onCancel(arguments: Any?) {
                events = null
            }
        })
    }

    fun handle(call: MethodCall, result: Result) = try {
        when (call.method) {
            "getPlatformVersion" -> result.success("Android ${android.os.Build.VERSION.RELEASE}")
            "evaluate" -> evaluate(result)
            "fit" -> fit(call, result)
            "getParameters" -> getParameters(result)
            "ready" -> ready(result)
            "testSize" -> result.success(flowerClient.testSamples.size)
            "trainingSize" -> result.success(flowerClient.trainingSamples.size)
            "updateParameters" -> updateParameters(call, result)
            "initML" -> initML(call, result)
            else -> result.notImplemented()
        }
    } catch (err: Throwable) {
        result.error(TAG, "$err", err.stackTraceToString())
    }

    fun evaluate(result: Result) = scope.launch(Dispatchers.Default) {
        val (loss, accuracy) = flowerClient.evaluate()
        val lossAccuracy = floatArrayOf(loss, accuracy)
        runOnUiThread { result.success(lossAccuracy) }
    }

    fun fit(call: MethodCall, result: Result) = scope.launch(Dispatchers.Default) {
        val epochs = call.argument<Int>("epochs")!!
        val batchSize = call.argument<Int>("batchSize")!!
        flowerClient.fit(epochs, batchSize) { runOnUiThread { events?.success(it) } }
        runOnUiThread { result.success(null) }
    }

    fun getParameters(result: Result) = flowerClient.getParameters().map { it.array() }.let {
        result.success(it)
    }

    fun ready(result: Result) =
        result.success(flowerClient.trainingSamples.isNotEmpty() && flowerClient.testSamples.isNotEmpty())

    fun updateParameters(call: MethodCall, result: Result) = scope.launch(Dispatchers.Default) {
        val parameters = call.argument<List<ByteArray>>("parameters")!!.map { ByteBuffer.wrap(it) }
            .toTypedArray()
        flowerClient.updateParameters(parameters)
        runOnUiThread { result.success(null) }
    }

    fun initML(call: MethodCall, result: Result) = scope.launch(Dispatchers.Default) {
        val dataDir = call.argument<String>("dataDir")!!
        val modelDir = call.argument<String>("modelDir")!!
        val layersSizes = call.argument<List<Int>>("layersSizes")!!.toIntArray()
        val partitionId = call.argument<Int>("partitionId")!!
        val buffer = loadMappedFile(File(modelDir))
        flowerClient = FlowerClient(buffer, layersSizes, sampleSpec())
        loadData(dataDir, flowerClient, partitionId)
        runOnUiThread { result.success(null) }
    }

    companion object {
        const val TAG = "MainActivity"
    }
}
