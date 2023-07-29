package org.eu.fedcampus.android_client

import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import org.eu.fedcampus.android_client.databinding.ActivityMainBinding
import org.eu.fedcampus.train.FlowerClient
import org.eu.fedcampus.train.Train
import org.eu.fedcampus.train.examples.cifar10.DATA_TYPE
import org.eu.fedcampus.train.examples.cifar10.Float3DArray
import org.eu.fedcampus.train.examples.cifar10.loadData
import org.eu.fedcampus.train.examples.cifar10.sampleSpec
import org.eu.fedcampus.train.helpers.deviceId
import org.eu.fedcampus.train.helpers.loadMappedFile
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private val scope = MainScope()
    lateinit var train: Train<Float3DArray, FloatArray>
    lateinit var flowerClient: FlowerClient<Float3DArray, FloatArray>
    private lateinit var binding: ActivityMainBinding
    val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.connectButton.setOnClickListener { connect() }
        binding.trainButton.setOnClickListener { startTrain() }
    }

    fun appendLog(text: String) {
        val time = dateFormat.format(Date())
        runOnUiThread {
            binding.logsTextView.append("\n$time   $text")
        }
    }

    fun connect() {
        val clientPartitionIdText = binding.clientPartitionIdEditText.text.toString()
        val flServerIpText = binding.flServerIpEditText.text.toString()
        val flServerPortText = binding.flServerPortEditText.text.toString()

        // Validate client partition id
        val partitionId: Int
        try {
            partitionId = clientPartitionIdText.toInt()
        } catch (e: NumberFormatException) {
            appendLog("Invalid client partition id!")
            return
        }

        // Validate backend server host
        val host: Uri
        try {
            host = Uri.parse("http://$flServerIpText")
            if (!host.path.isNullOrEmpty() || host.host.isNullOrEmpty()) {
                throw Exception()
            }
        } catch (e: Exception) {
            appendLog("Invalid backend server host!")
            return
        }

        // Validate backend server port
        val backendPort: Int
        val backendUrl: Uri
        try {
            backendPort = flServerPortText.toInt()
            backendUrl = Uri.parse("http://$flServerIpText:$backendPort")

        } catch (e: NumberFormatException) {
            appendLog("Invalid backend server port!")
            return
        }

        appendLog("Connecting with Partition ID: $partitionId, Server IP: $host, Port: $backendPort")

        scope.launch {
            try {
                connectInBackground(partitionId, backendUrl, host)
            } catch (err: Throwable) {
                appendLog("$err")
                Log.e(TAG, err.stackTraceToString())
                runOnUiThread { binding.connectButton.isEnabled = true }
            }
        }
        binding.connectButton.isEnabled = false
        appendLog("Creating channel object.")
    }

    fun startTrain() {
        scope.launch {
            try {
                trainInBackground()
            } catch (err: Throwable) {
                appendLog("$err")
                Log.e(TAG, err.stackTraceToString())
                binding.trainButton.isEnabled = true
            }
        }
    }

    @Throws
    suspend fun connectInBackground(participationId: Int, backendUrl: Uri, host: Uri) {
        Log.i(TAG, "Backend URL: $backendUrl")
        train = Train(this, backendUrl.toString(), sampleSpec())
        train.enableTelemetry(deviceId(this))
        val modelFile = train.prepareModel(DATA_TYPE)
        val serverData = train.getServerInfo(binding.startFreshCheckBox.isChecked)
        if (serverData.port == null) {
            throw Error("Flower server port not available, status ${serverData.status}")
        }
        flowerClient = train.prepare(
            loadMappedFile(modelFile), "dns:///${host.host}:${serverData.port}", false
        )
        loadData(this, flowerClient, participationId)

        appendLog("Connected to Flower server on port ${serverData.port} and loaded data set.")
        runOnUiThread {
            binding.trainButton.isEnabled = true
        }
    }

    fun trainInBackground() {
        train.start {
            runOnUiThread { appendLog(it) }
        }
        appendLog("Started training.")
        runOnUiThread {
            binding.trainButton.isEnabled = false
        }
    }
}

private const val TAG = "MainActivity"
