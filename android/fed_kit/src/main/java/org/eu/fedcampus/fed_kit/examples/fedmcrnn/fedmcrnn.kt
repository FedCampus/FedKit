package org.eu.fedcampus.fed_kit.examples.fedmcrnn

import org.eu.fedcampus.fed_kit_train.SampleSpec
import org.eu.fedcampus.fed_kit_train.helpers.maxSquaredErrorLoss
import org.eu.fedcampus.fed_kit_train.helpers.placeholderAccuracy

fun sampleSpec() = SampleSpec<Float2DArray, FloatArray>(
    { it.toTypedArray() },
    { it.toTypedArray() },
    { Array(it) { FloatArray(1) } },
    ::maxSquaredErrorLoss,
    ::placeholderAccuracy,
)

const val DATA_TYPE = "FedMCRNN_7x8"

typealias Float2DArray = Array<FloatArray>
