package org.eu.fedcampus.fed_kit

import io.grpc.ManagedChannel
import org.eu.fedcampus.fed_kit_train.db.TFLiteModel
import org.eu.fedcampus.fed_kit_train.FlowerClient

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
