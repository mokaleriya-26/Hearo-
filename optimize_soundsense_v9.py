import os
import numpy as np
import tensorflow as tf

BASE = "soundsense_padded_model_v9_recording_f1"
SOURCE = os.path.join(BASE, "best_model_v9_recording_macro_f1.keras")

OUTDIR = "soundsense_deployment/v9/fusion_experiment"
os.makedirs(OUTDIR, exist_ok=True)

FUSED_KERAS = os.path.join(OUTDIR, "soundsense_v9_fused.keras")
FUSED_TFLITE = os.path.join(OUTDIR, "soundsense_v9_fused_float32.tflite")

print("TensorFlow:", tf.__version__)
print("Loading original V9 model...")
original = tf.keras.models.load_model(SOURCE, compile=False)

def fold_conv_bn(conv, bn):
    """Fold an inference-mode BatchNormalization layer into Conv2D."""
    weights = conv.get_weights()

    if len(weights) == 2:
        kernel, conv_bias = weights
    elif len(weights) == 1:
        kernel = weights[0]
        conv_bias = np.zeros(kernel.shape[-1], dtype=np.float32)
    else:
        raise ValueError(
            f"Unexpected weights in {conv.name}: {len(weights)} arrays"
        )

    gamma, beta, moving_mean, moving_variance = bn.get_weights()

    if not bn.scale:
        gamma = np.ones_like(moving_mean)
    if not bn.center:
        beta = np.zeros_like(moving_mean)

    scale = gamma / np.sqrt(moving_variance + bn.epsilon)

    fused_kernel = kernel * scale.reshape((1, 1, 1, -1))
    fused_bias = (conv_bias - moving_mean) * scale + beta

    return fused_kernel, fused_bias

# Build a new graph with the same input and output behavior.
inputs = tf.keras.Input(shape=original.input_shape[1:], name="input_layer")
x = inputs

pairs = [
    ("conv2d", "batch_normalization"),
    ("conv2d_1", "batch_normalization_1"),
    ("conv2d_2", "batch_normalization_2"),
]

fused_weights = {}

for conv_name, bn_name in pairs:
    conv = original.get_layer(conv_name)
    bn = original.get_layer(bn_name)

    fused_weights[conv_name] = fold_conv_bn(conv, bn)

    # Copy convolution configuration, enabling bias after folding BN.
    config = conv.get_config()
    config["use_bias"] = True
    new_conv = tf.keras.layers.Conv2D.from_config(config)

    x = new_conv(x)
    x = original.get_layer(bn_name) if False else x

    # Preserve the activation and pooling order. The original Conv2D
    # activation is applied inside the Conv2D layer.
    new_conv._name = conv_name + "_fused"

    # Set weights immediately; layer is already built after symbolic call.
    new_conv.set_weights(list(fused_weights[conv_name]))

    if conv_name == "conv2d":
        x = original.get_layer("max_pooling2d")(x)
    elif conv_name == "conv2d_1":
        x = original.get_layer("max_pooling2d_1")(x)

# Third fused convolution is followed by global average pooling.
x = original.get_layer("global_average_pooling2d")(x)
x = original.get_layer("dense")(x)
# Dropout is inactive during inference; preserve it in the graph.
x = original.get_layer("dropout")(x)
x = original.get_layer("dense_1")(x)

fused = tf.keras.Model(inputs, x, name="soundsense_v9_bn_fused")

# Check the fused model against the original using deterministic input.
rng = np.random.default_rng(42)
test_input = rng.normal(size=(2, 201, 13, 1)).astype(np.float32)

reference = original(test_input, training=False).numpy()
candidate = fused(test_input, training=False).numpy()

max_abs = float(np.max(np.abs(reference - candidate)))
mean_abs = float(np.mean(np.abs(reference - candidate)))

print("\n===== PREDICTION COMPARISON =====")
print("Original output shape:", reference.shape)
print("Fused output shape:   ", candidate.shape)
print("Maximum absolute difference:", max_abs)
print("Mean absolute difference:   ", mean_abs)

if not np.allclose(reference, candidate, rtol=2e-4, atol=2e-5):
    print("\nERROR: Fused model did not match the original closely enough.")
    print("Not saving or converting this candidate.")
    raise SystemExit(1)

fused.save(FUSED_KERAS)
print("\nSaved fused Keras model:", FUSED_KERAS)

converter = tf.lite.TFLiteConverter.from_keras_model(fused)
converter.optimizations = []
tflite_bytes = converter.convert()

with open(FUSED_TFLITE, "wb") as f:
    f.write(tflite_bytes)

print("Saved fused TFLite model:", FUSED_TFLITE)
print("TFLite file size:", len(tflite_bytes), "bytes")

# Inspect operators and output agreement after conversion.
interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

# The converted model may have a fixed batch size of 1.
single_test_input = test_input[:1]

interpreter.set_tensor(
    input_detail["index"],
    single_test_input
)
interpreter.invoke()

tflite_output = interpreter.get_tensor(
    output_detail["index"]
)

reference_single = reference[:1]

print("\n===== TFLITE VERIFICATION =====")
print("Input shape:", input_detail["shape"])
print("Input dtype:", input_detail["dtype"])
print("Output shape:", output_detail["shape"])
print("Output dtype:", output_detail["dtype"])

print(
    "Maximum difference vs original:",
    float(np.max(np.abs(reference_single - tflite_output)))
)

print("Operators:", [
    op["op_name"]
    for op in interpreter._get_ops_details()
    if op["op_name"] != "DELEGATE"
])

print("\nExperiment complete. Original files were not modified.")

print("\n===== TFLITE VERIFICATION =====")
print("Input shape:", input_detail["shape"])
print("Input dtype:", input_detail["dtype"])
print("Output shape:", output_detail["shape"])
print("Output dtype:", output_detail["dtype"])
print("Maximum difference vs original:",
      float(np.max(np.abs(reference - tflite_output))))
print("Operators:", [
    op["op_name"] for op in interpreter._get_ops_details()
    if op["op_name"] != "DELEGATE"
])

print("\nExperiment complete. Original files were not modified.")
