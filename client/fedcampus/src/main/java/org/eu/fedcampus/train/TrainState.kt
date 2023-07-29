package org.eu.fedcampus.train

import io.grpc.ManagedChannel
import org.eu.fedcampus.train.db.TFLiteModel

sealed class TrainState<X : Any, Y : Any> {
    class Initialized<X : Any, Y : Any> : TrainState<X, Y>()

    data class WithModel<X : Any, Y : Any>(val model: TFLiteModel) : TrainState<X, Y>()

    data class Prepared<X : Any, Y : Any>(
        val model: TFLiteModel,
        val flowerClient: FlowerClient<X, Y>,
        val channel: ManagedChannel
    ) : TrainState<X, Y>()

    data class Training<X : Any, Y : Any>(
        val model: TFLiteModel,
        val flowerClient: FlowerClient<X, Y>,
        val flowerServiceRunnable: FlowerServiceRunnable<X, Y>
    ) : TrainState<X, Y>()
}
