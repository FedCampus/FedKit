package org.eu.fedcampus.fed_kit.helpers

import org.eu.fedcampus.fed_kit.Sample

/** Zero-one accuracy for one-hot classifier. */
fun <X> classifierAccuracy(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    sample.label[logit.argmax()]
}

/** Signifies that "accuracy" makes no sense for the model. */
@Suppress("UNUSED_PARAMETER")
fun <X, Y> placeholderAccuracy(
    samples: MutableList<Sample<X, Y>>,
    logits: Array<Y>
): Float = -1f
