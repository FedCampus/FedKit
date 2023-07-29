package org.eu.fedcampus.train.helpers

import android.annotation.SuppressLint
import android.content.Context
import android.provider.Settings
import android.util.Log
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

@Throws(IOException::class)
fun loadMappedFile(file: File): MappedByteBuffer {
    Log.i("Loading mapped file", "$file")
    val accessFile = RandomAccessFile(file, "r")
    val channel = accessFile.channel
    return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
}

@Throws(IOException::class)
fun loadMappedAssetFile(context: Context, filePath: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(filePath)
    val fileChannel = fileDescriptor.createInputStream().channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

infix fun <T, R> Iterable<T>.lazyZip(other: Array<out R>): Sequence<Pair<T, R>> {
    val ours = iterator()
    val theirs = other.iterator()

    return sequence {
        while (ours.hasNext() && theirs.hasNext()) {
            yield(ours.next() to theirs.next())
        }
    }
}

fun <T, R> Iterable<T>.lazyMap(transform: (T) -> R) = sequence {
    iterator().forEach { yield(transform(it)) }
}

fun FloatArray.argmax(): Int = indices.maxBy { this[it] }

fun stringToLong(string: String): Long {
    val hashCode = string.hashCode().toLong()
    val secondHashCode = string.reversed().hashCode().toLong()
    return (hashCode shl 32) or secondHashCode
}

@SuppressLint("HardwareIds")
fun deviceId(context: Context): Long {
    val androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
    return stringToLong(androidId)
}

@Throws(AssertionError::class)
fun assertIntsEqual(expected: Int, actual: Int) {
    if (expected != actual) {
        throw AssertionError("Test failed: expected `$expected`, got `$actual` instead.")
    }
}
