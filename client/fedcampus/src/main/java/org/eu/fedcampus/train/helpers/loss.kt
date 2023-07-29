package org.eu.fedcampus.train.helpers

import org.eu.fedcampus.train.Sample
import kotlin.math.ln

fun <X> negativeLogLikelihoodLoss(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    -ln(logit[sample.label.argmax()])
}

fun <X> maxSquaredErrorLoss(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    sample.label.zip(logit).fold(0f) { acc, (real, prediction) ->
        val diff = real - prediction
        diff * diff + acc
    }
}

fun <X, Y> averageLossWith(
    samples: MutableList<Sample<X, Y>>,
    logits: Array<Y>,
    loss: (Sample<X, Y>, logit: Y) -> Float
) = if (samples.isEmpty()) 0f else {
    var lossSum = 0f
    for ((sample, logit) in samples lazyZip logits) {
        lossSum += loss(sample, logit)
    }
    lossSum / samples.size
}
