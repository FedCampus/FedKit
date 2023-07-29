package org.eu.fedcampus.train.examples.cifar10

import org.eu.fedcampus.train.SampleSpec
import org.eu.fedcampus.train.helpers.classifierAccuracy
import org.eu.fedcampus.train.helpers.negativeLogLikelihoodLoss

fun sampleSpec() = SampleSpec<Float3DArray, FloatArray>(
    { it.toTypedArray() },
    { it.toTypedArray() },
    { Array(it) { FloatArray(CLASSES.size) } },
    ::negativeLogLikelihoodLoss,
    ::classifierAccuracy,
)

const val DATA_TYPE = "CIFAR10_32x32x3"

typealias Float3DArray = Array<Array<FloatArray>>
