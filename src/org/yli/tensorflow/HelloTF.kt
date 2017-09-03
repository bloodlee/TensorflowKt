package org.yli.tensorflow

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import java.nio.charset.Charset

fun main(args: Array<String>) {
    val g = Graph()

    val value = "Hello from ${TensorFlow.version()}"

    val charset = Charset.forName("UTF-8")

    val t = Tensor.create(value.toByteArray(charset))
    g.opBuilder("Const", "MyConst")
            .setAttr("dtype", t.dataType())
            .setAttr("value", t)
            .build()

    val t1 = Tensor.create(1)

    val s = Session(g)
    val output = s.runner().fetch("MyConst").run().get(0)

    println(String(output.bytesValue(), charset))
}

