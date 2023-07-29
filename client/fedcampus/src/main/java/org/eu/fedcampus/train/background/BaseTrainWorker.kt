package org.eu.fedcampus.train.background

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ForegroundInfo
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.eu.fedcampus.train.FlowerClient
import org.eu.fedcampus.train.SampleSpec
import org.eu.fedcampus.train.Train
import org.eu.fedcampus.train.helpers.loadMappedFile
import java.util.concurrent.TimeUnit

/**
 * Inherit your training worker from this and fill in the constructor parameters
 * (except for [context] and `params`) with concrete values.
 * @param icon Small icon for the notification.
 */
open class BaseTrainWorker<X : Any, Y : Any>(
    val context: Context, params: WorkerParameters,
    val icon: Int,
    val sampleSpec: SampleSpec<X, Y>,
    val dataType: String,
    val loadData: suspend (Context, FlowerClient<X, Y>, Int) -> Unit,
    val trainCallback: (String) -> Unit,
    val useTLS: Boolean = false,
) : CoroutineWorker(context, params) {
    val data = inputData
    lateinit var train: Train<X, Y>

    override suspend fun doWork() = try {
        setForeground(ForegroundInfo(1, createNotification("Training", context, icon)))
        train()
    } catch (err: Throwable) {
        Log.e(TAG, err.stackTraceToString())
        Result.retry()
    }

    suspend fun train(): Result {
        val backendUrl = data.getString("backendUrl")!!
        val deviceId = data.getLong("deviceId", 0L)
        val flowerHost = data.getString("flowerHost")!!
        val participantId = data.getInt("participantId", 1)

        train = Train(context, backendUrl, sampleSpec)
        if (deviceId != 0L) train.enableTelemetry(deviceId)
        Log.i(TAG, "Starting with backend $backendUrl for $dataType.")

        val flowerClient = prepare(flowerHost)
        Log.i(TAG, "Prepared $flowerClient.")

        loadData(context, flowerClient, participantId)
        Log.i(TAG, "Loaded data.")

        train.start(trainCallback).use {
            Log.i(TAG, "Training.")
            withContext(Dispatchers.IO) {
                it.finishLatch.await()
            }
        }
        Log.i(TAG, "Finished.")

        return Result.success()
    }

    private suspend fun prepare(flowerHost: String): FlowerClient<X, Y> {
        val modelFile = train.prepareModel(dataType)
        val serverData = train.getServerInfo()
        if (serverData.port == null) {
            throw Error("Flower server port not available, status ${serverData.status}")
        }
        val address = "dns:///$flowerHost:${serverData.port}"
        return train.prepare(loadMappedFile(modelFile), address, useTLS)
    }

    companion object {
        const val TAG = "TrainWorker"
    }
}

fun trainWorkerData(backendUrl: String, deviceId: Long, flowerHost: String, participantId: Int) =
    Data.Builder().putString("backendUrl", backendUrl).putLong("deviceId", deviceId)
        .putString("flowerHost", flowerHost).putInt("participantId", participantId).build()

@Suppress("unused")
inline fun <reified W : BaseTrainWorker<X, Y>, X : Any, Y : Any> trainWorkRequest(inputData: Data) =
    PeriodicWorkRequestBuilder<W>(
        1, TimeUnit.HOURS,
        50, TimeUnit.MINUTES,
    ).setConstraints(realIdleConstraints()).setInputData(inputData).addTag(BaseTrainWorker.TAG)
        .build()

inline fun <reified W : BaseTrainWorker<X, Y>, X : Any, Y : Any> fastTrainWorkRequest(inputData: Data) =
    PeriodicWorkRequestBuilder<W>(
        15, TimeUnit.MINUTES, // Minimum interval allowed.
        10, TimeUnit.MINUTES,
    ).setConstraints(wifiConstraints()).setInputData(inputData).addTag(BaseTrainWorker.TAG).build()

fun realIdleConstraints() =
    Constraints.Builder().setRequiredNetworkType(NetworkType.UNMETERED).setRequiresCharging(true)
        .setRequiresDeviceIdle(true).build()

fun wifiConstraints() = Constraints.Builder().setRequiredNetworkType(NetworkType.UNMETERED).build()

/**
 * Create a Notification that is shown as a heads-up notification if possible.
 *
 * Adopted from `codelab-android-workmanager`.
 */
fun createNotification(message: String, context: Context, icon: Int): Notification {
    // Make a channel if necessary
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        // Create the NotificationChannel, but only on API 26+ because
        // the NotificationChannel class is new and not in the support library
        val name = "FedCampus"
        val importance = NotificationManager.IMPORTANCE_HIGH
        val channel = NotificationChannel(CHANNEL_ID, name, importance)

        // Add the channel
        val notificationManager =
            context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager?

        notificationManager?.createNotificationChannel(channel)
    }

    // Create the notification
    return NotificationCompat.Builder(context, CHANNEL_ID)
        .setSmallIcon(icon)
        .setContentTitle("FedCampus")
        .setContentText(message)
        .setPriority(NotificationCompat.PRIORITY_HIGH)
        .setVibrate(LongArray(0)).build()
}

const val CHANNEL_ID = "FedCampus Channel"
