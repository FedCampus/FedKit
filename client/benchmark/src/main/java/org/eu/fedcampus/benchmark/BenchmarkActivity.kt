package org.eu.fedcampus.benchmark

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.util.Patterns
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.webkit.URLUtil
import androidx.appcompat.app.AppCompatActivity
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.WorkManager
import org.eu.fedcampus.benchmark.databinding.ActivityBenchmarkBinding
import org.eu.fedcampus.train.background.BaseTrainWorker
import org.eu.fedcampus.train.background.fastTrainWorkRequest
import org.eu.fedcampus.train.background.trainWorkerData
import org.eu.fedcampus.train.examples.cifar10.Float3DArray
import org.eu.fedcampus.train.helpers.deviceId
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class BenchmarkActivity : AppCompatActivity() {
    lateinit var binding: ActivityBenchmarkBinding
    lateinit var workManager: WorkManager
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityBenchmarkBinding.inflate(layoutInflater)
        setContentView(binding.root)

        workManager = WorkManager.getInstance(this)

        enterClickSub()
        binding.btSub.setOnClickListener { submit() }

        appendLog("Initialized.")
    }

    private fun submit() {
        hideKeyboard()
        val uri = binding.etUri.text.toString()
        val url = "http://$uri:8000"
        if (!isUrl(url)) {
            appendLog("$url is invalid, please input a valid URI!")
            return
        }

        val inputData = trainWorkerData(url, deviceId(this), uri, 1)
        val trainWork =
            fastTrainWorkRequest<BenchmarkCifar10Worker, Float3DArray, FloatArray>(inputData)
        workManager.enqueueUniquePeriodicWork(
            BaseTrainWorker.TAG,
            ExistingPeriodicWorkPolicy.REPLACE,
            trainWork
        )
        appendLog("Submit training work request for $uri")
    }

    private fun enterClickSub() = binding.etUri.setOnEditorActionListener { _, actionId, _ ->
        if (actionId == EditorInfo.IME_ACTION_DONE) {
            binding.btSub.performClick()
            return@setOnEditorActionListener true
        }
        false
    }

    private fun appendLog(log: String) {
        val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val time = dateFormat.format(Date())
        val text = "$time: $log\n"
        Log.i(TAG, "appendLog: $text")
        runOnUiThread {
            binding.tvHello.append(text)
        }
    }

    @SuppressLint("ServiceCast")
    private fun hideKeyboard() {
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        var view = currentFocus
        if (view == null) {
            view = View(this)
        }
        imm.hideSoftInputFromWindow(view.windowToken, 0)
    }

    companion object {
        const val TAG = "BenchmarkActivity"
    }
}

fun isUrl(url: String) = URLUtil.isValidUrl(url) && Patterns.WEB_URL.matcher(url).matches()
