package org.eu.fedcampus.fed_kit.examples.cifar10

import org.eu.fedcampus.fed_kit_train.SampleSpec
import org.eu.fedcampus.fed_kit_train.helpers.classifierAccuracy
import org.eu.fedcampus.fed_kit_train.helpers.negativeLogLikelihoodLoss

fun sampleSpec() = SampleSpec<Float3DArray, FloatArray>(
    { it.toTypedArray() },
    { it.toTypedArray() },
    { Array(it) { FloatArray(CLASSES.size) } },
    ::negativeLogLikelihoodLoss,
    ::classifierAccuracy,
)

const val DATA_TYPE = "CIFAR10_32x32x3"

typealias Float3DArray = Array<Array<FloatArray>>
