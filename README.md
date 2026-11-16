# Plakami: licence plate and vehicle detector

*Plakami* combines Turkish *plaka* (licence plate) and Japanese *mi* (見, to see).

A small detector (YOLOX-Tiny, about 5 M parameters) that finds **licence plates** and
**vehicles** in a photo. It returns a box, a class and a confidence score for each one,
for example so your app can **blur plates** before a photo is saved or shared.

- **Phones:** `detector_fp16.tflite` (9.7 MB), via LiteRT / TensorFlow Lite on Android and iOS.
- **Servers and desktops:** `detector.onnx` (19.3 MB), via ONNX Runtime.
- **Licence:** MIT. The model derives from YOLOX (Apache-2.0); training images are credited
  in `ATTRIBUTION.md`.

This guide covers integration. Accuracy, intended use and limits are in `MODEL_CARD.md`.

## What's in a release

| File | What it is |
|---|---|
| `detector_fp16.tflite` | Phone model. Input NHWC `(1, 640, 640, 3)`, output `(1, 8400, 7)`, both float32 |
| `detector.onnx` | Server/desktop model. Input NCHW `(1, 3, 640, 640)` named `images`, output `(1, 8400, 7)` |
| `labels.txt` | Class names in output order: `license_plate`, `vehicle` |
| `release.json` | The exact contract: input format, output layout, thresholds, tiling, metrics, file hashes |
| `postprocess.py` | Reference Python code for every step below (NumPy + Pillow only) |
| `test_vectors/` | Five photos with the exact expected outputs, to check your port |
| `MODEL_CARD.md` | Intended use, accuracy, limits, training data |
| `SHA256SUMS` | Checksums; verify with `shasum -a 256 -c SHA256SUMS` |

Both models take the same pixel values and give the same output; only the input layout
differs. They contain no pre- or post-processing, so **your app does steps 1–7 below**.

## Quick start (Python)

```python
import onnxruntime as ort
from PIL import Image, ImageOps

import postprocess as pp  # from the release

session = ort.InferenceSession("detector.onnx", providers=["CPUExecutionProvider"])
THRESHOLDS = {1: 0.10, 2: 0.30}  # 1 = license_plate, 2 = vehicle


def detect(image):
    tensor, ratio = pp.letterbox(image, 640, "bgr")
    raw = session.run(None, {"images": tensor})[0]
    return pp.postprocess(
        raw,
        size=640,
        ratio=ratio,
        image_width=image.width,
        image_height=image.height,
        conf_threshold=0.10,
        nms_iou=0.45,
    )


photo = ImageOps.exif_transpose(Image.open("car.jpg")).convert("RGB")
boxes = [
    d
    for d in pp.detect_tiled(photo, detect, grid=2, overlap=0.2)
    if d.score >= THRESHOLDS[d.class_id]
]
for d in boxes:
    print(d.class_id, round(d.score, 2), d.x1, d.y1, d.x2, d.y2)  # original-photo pixels
```

## How it works: steps for any language

Numbers below are for this release (`release.json` has them all). `postprocess.py` is the
reference: when in doubt, it is right.

### 1. Load the photo upright

Apply the EXIF orientation first, so the pixels are the photo as the user sees it. Phone
cameras often store photos sideways with an orientation tag. On Android, read the tag with
`ExifInterface` and rotate the bitmap. On iOS, redraw a `UIImage` whose `imageOrientation`
isn't `.up` before taking its `cgImage`. Boxes come back in the pixels of this upright image.

### 2. Letterbox to 640 × 640

```
ratio  = min(640 / width, 640 / height)
new_w  = max(1, floor(width  * ratio))     # truncate, don't round
new_h  = max(1, floor(height * ratio))
resized = bilinear_resize(photo, new_w, new_h)
canvas  = 640 x 640, every pixel (114, 114, 114)
paste resized at the TOP-LEFT corner (x = 0, y = 0); don't centre it
```

- Compute `ratio` in double precision and keep it for step 6.
- The reference resize is bilinear with OpenCV `INTER_LINEAR` semantics. Platform resizers
  (`Bitmap.createScaledBitmap`, Core Graphics) differ by a grey level or two. That is fine,
  but expect scores to differ by up to about 0.01 from the test vectors.

### 3. Build the input tensor

- **Channel order is BGR**, not RGB.
- **Values are raw 0–255 as float32.** No division by 255, no mean or std.
- **TFLite (NHWC):** for each row, for each pixel: B, G, R. Shape `(1, 640, 640, 3)`.
- **ONNX (NCHW):** the whole B plane, then G, then R. Shape `(1, 3, 640, 640)`, input `images`.

### 4. Run the model

The output has shape `(1, 8400, 7)`. Each row has these 7 columns:

```
raw_cx, raw_cy, raw_w, raw_h, objectness, p_license_plate, p_vehicle
```

The last three are already probabilities (0–1); don't apply a sigmoid again. The box
columns are **not decoded yet**.

### 5. Decode the boxes and scores

The 8,400 rows are three grids in this order: stride 8 (80 × 80 = 6,400 rows), stride 16
(40 × 40 = 1,600), stride 32 (20 × 20 = 400). Within each grid, rows go row by row (`gy`),
then cell by cell (`gx`).

```
row = 0
for stride in [8, 16, 32]:
    cells = 640 / stride
    for gy in 0 ..< cells:
        for gx in 0 ..< cells:
            r = output[row]; row += 1
            cls   = 0 if r[5] >= r[6] else 1          # 0 = license_plate, 1 = vehicle
            score = r[4] * r[5 + cls]
            if score < threshold[cls]: continue       # plate 0.10, vehicle 0.30
            cx = (r[0] + gx) * stride;  cy = (r[1] + gy) * stride
            w  = exp(r[2]) * stride;    h  = exp(r[3]) * stride
            box = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)   # 640 x 640 input pixels
```

Class indexes here are 0-based, matching `labels.txt`. `postprocess.py` reports them
1-based (`1 = license_plate`, `2 = vehicle`).

### 6. Non-maximum suppression, then back to photo pixels

- **NMS per class:** sort that class's boxes by score, highest first; ties keep their
  original order. Keep the top box, drop every remaining box whose IoU with it is above
  **0.45**, and repeat.
- **Back to photo pixels:** divide each coordinate by `ratio`, then clip x to `[0, width]`
  and y to `[0, height]`.

### 7. Tiling (recommended): whole photo + 2 × 2 tiles

Small, distant plates are often too small after shrinking the whole photo to 640 px.
Running 4 overlapping tiles as well raises small-plate recall a lot (see `MODEL_CARD.md`).
It costs 5 model runs per photo, about 0.45 s on a laptop CPU with the FP16 model.

```
spans(dim):                                   # one axis; grid = 2, overlap = 0.2
    tile = min(dim, ceil(dim / 1.8))          # 1.8 = grid - (grid - 1) * overlap
    return [(0, tile), (dim - tile, dim)]

detections = steps 2-6 on the whole photo
for (y1, y2) in spans(height):
    for (x1, x2) in spans(width):             # 4 tiles
        tile = crop(photo, x1, y1, x2, y2)
        mx = max(2, 0.02 * (x2 - x1));  my = max(2, 0.02 * (y2 - y1))
        for d in steps 2-6 on tile:           # d is in tile pixels
            cut = (x1 > 0      and d.x1 <= mx) or (y1 > 0       and d.y1 <= my) or
                  (x2 < width  and d.x2 >= (x2 - x1) - mx) or
                  (y2 < height and d.y2 >= (y2 - y1) - my)
            if not cut:
                detections.add(d shifted by (x1, y1))
result = NMS per class (IoU 0.45) over detections
```

- The `cut` rule drops boxes touching an inner tile edge: they are pieces of an object that
  the whole-photo pass or the next tile sees whole.
- Apply the class thresholds inside each pass (step 5), as shown. That gives the same
  result as the reference, which filters after merging.

### 8. Blur the plates

The model only finds boxes; your app does the blurring.

- Blur on the **full-resolution** photo, using the boxes from step 6 or 7.
- Pad each box by about 10% on every side, since boxes can be a few pixels tight.
- Use a strong effect (pixelate to about 8 blocks across the plate, or a heavy Gaussian
  blur) so the characters can't be recovered.
- Keep the low plate threshold (0.10). A false box only blurs a harmless patch; a missed
  plate leaks.

## Android (Kotlin, LiteRT / TensorFlow Lite)

> **Not compiled.** These are sketches of the steps above. Before shipping, check your port
> with the test vectors (see "Check your port").

Add LiteRT (`com.google.ai.edge.litert:litert`) or `org.tensorflow:tensorflow-lite`. Put
`detector_fp16.tflite` in `assets/` and memory-map it.

```kotlin
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** Box in photo pixels. classId: 0 = license_plate, 1 = vehicle (labels.txt order). */
data class Detection(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val score: Float, val classId: Int,
)

class PlateDetector(model: MappedByteBuffer, threads: Int = 4) {
    private val size = 640
    private val thresholds = floatArrayOf(0.10f, 0.30f)   // plate, vehicle
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(threads))
    private val input = ByteBuffer.allocateDirect(4 * size * size * 3).order(ByteOrder.nativeOrder())
    private val output = Array(1) { Array(8400) { FloatArray(7) } }

    /** Whole photo + 2x2 tiles (step 7). `photo` must already be upright (step 1). */
    fun detect(photo: Bitmap): List<Detection> {
        val w = photo.width
        val h = photo.height
        val all = ArrayList(detectOnce(photo))
        for ((y1, y2) in spans(h)) for ((x1, x2) in spans(w)) {
            val tile = Bitmap.createBitmap(photo, x1, y1, x2 - x1, y2 - y1)
            val mx = max(2f, 0.02f * (x2 - x1))
            val my = max(2f, 0.02f * (y2 - y1))
            for (d in detectOnce(tile)) {
                val cut = (x1 > 0 && d.x1 <= mx) || (y1 > 0 && d.y1 <= my) ||
                    (x2 < w && d.x2 >= (x2 - x1) - mx) || (y2 < h && d.y2 >= (y2 - y1) - my)
                if (!cut) all += d.copy(x1 = d.x1 + x1, y1 = d.y1 + y1, x2 = d.x2 + x1, y2 = d.y2 + y1)
            }
        }
        return nms(all)
    }

    private fun spans(dim: Int): List<Pair<Int, Int>> {
        val tile = min(dim, ceil(dim / 1.8).toInt())
        return listOf(0 to tile, (dim - tile) to dim)
    }

    /** Steps 2-6 on one image. */
    fun detectOnce(bitmap: Bitmap): List<Detection> {
        val ratio = min(size.toDouble() / bitmap.width, size.toDouble() / bitmap.height)
        val nw = max(1, (bitmap.width * ratio).toInt())
        val nh = max(1, (bitmap.height * ratio).toInt())
        val resized = Bitmap.createScaledBitmap(bitmap, nw, nh, true)   // bilinear
        val pixels = IntArray(nw * nh)
        resized.getPixels(pixels, 0, nw, 0, 0, nw, nh)

        input.rewind()
        for (y in 0 until size) for (x in 0 until size) {
            if (x < nw && y < nh) {
                val p = pixels[y * nw + x]                 // ARGB
                input.putFloat((p and 0xFF).toFloat())          // B
                input.putFloat((p shr 8 and 0xFF).toFloat())    // G
                input.putFloat((p shr 16 and 0xFF).toFloat())   // R
            } else {
                input.putFloat(114f); input.putFloat(114f); input.putFloat(114f)
            }
        }
        input.rewind()
        interpreter.run(input, output)

        val found = ArrayList<Detection>()
        var row = 0
        for (stride in intArrayOf(8, 16, 32)) {
            val cells = size / stride
            for (gy in 0 until cells) for (gx in 0 until cells) {
                val r = output[0][row++]
                val cls = if (r[5] >= r[6]) 0 else 1
                val score = r[4] * r[5 + cls]
                if (score < thresholds[cls]) continue
                val cx = (r[0] + gx) * stride
                val cy = (r[1] + gy) * stride
                val bw = exp(r[2]) * stride
                val bh = exp(r[3]) * stride
                found += Detection(
                    ((cx - bw / 2) / ratio).toFloat().coerceIn(0f, bitmap.width.toFloat()),
                    ((cy - bh / 2) / ratio).toFloat().coerceIn(0f, bitmap.height.toFloat()),
                    ((cx + bw / 2) / ratio).toFloat().coerceIn(0f, bitmap.width.toFloat()),
                    ((cy + bh / 2) / ratio).toFloat().coerceIn(0f, bitmap.height.toFloat()),
                    score, cls,
                )
            }
        }
        return nms(found)
    }

    private fun nms(boxes: List<Detection>, iouThreshold: Float = 0.45f): List<Detection> {
        val kept = ArrayList<Detection>()
        for (cls in boxes.map { it.classId }.distinct().sorted()) {
            var order = boxes.filter { it.classId == cls }.sortedByDescending { it.score }
            while (order.isNotEmpty()) {
                val best = order.first()
                kept += best
                order = order.drop(1).filter { iou(best, it) <= iouThreshold }
            }
        }
        return kept.sortedByDescending { it.score }
    }

    private fun iou(a: Detection, b: Detection): Float {
        val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = iw * ih
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (union > 0f) inter / union else 0f
    }
}
```

Run `detect` off the main thread. One `Interpreter` isn't thread-safe; use one per thread.

## iOS (Swift, TensorFlow Lite)

> **Not compiled.** A sketch of the steps above. Before shipping, check your port with the
> test vectors.

Add the `TensorFlowLiteSwift` pod (or the LiteRT Swift package) and bundle
`detector_fp16.tflite`.

```swift
import CoreGraphics
import Foundation
import TensorFlowLite

/// Box in photo pixels. classId: 0 = license_plate, 1 = vehicle (labels.txt order).
struct Detection {
    var x1, y1, x2, y2: Float
    var score: Float
    var classId: Int
}

final class PlateDetector {
    private let size = 640
    private let thresholds: [Float] = [0.10, 0.30]   // plate, vehicle
    private let interpreter: Interpreter

    init(modelPath: String, threads: Int = 4) throws {
        var options = Interpreter.Options()
        options.threadCount = threads
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()
    }

    /// Whole photo + 2x2 tiles (step 7). `photo` must already be upright (step 1).
    func detect(_ photo: CGImage) throws -> [Detection] {
        let w = photo.width, h = photo.height
        var all = try detectOnce(photo)
        for (y1, y2) in spans(h) {
            for (x1, x2) in spans(w) {
                let rect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
                guard let tile = photo.cropping(to: rect) else { continue }
                let mx = max(2, 0.02 * Float(x2 - x1)), my = max(2, 0.02 * Float(y2 - y1))
                for d in try detectOnce(tile) {
                    let cut = (x1 > 0 && d.x1 <= mx) || (y1 > 0 && d.y1 <= my)
                        || (x2 < w && d.x2 >= Float(x2 - x1) - mx)
                        || (y2 < h && d.y2 >= Float(y2 - y1) - my)
                    if !cut {
                        all.append(Detection(x1: d.x1 + Float(x1), y1: d.y1 + Float(y1),
                                             x2: d.x2 + Float(x1), y2: d.y2 + Float(y1),
                                             score: d.score, classId: d.classId))
                    }
                }
            }
        }
        return nms(all)
    }

    private func spans(_ dim: Int) -> [(Int, Int)] {
        let tile = min(dim, Int((Double(dim) / 1.8).rounded(.up)))
        return [(0, tile), (dim - tile, dim)]
    }

    /// Steps 2-6 on one image.
    func detectOnce(_ image: CGImage) throws -> [Detection] {
        let ratio = min(Double(size) / Double(image.width), Double(size) / Double(image.height))
        let nw = max(1, Int(Double(image.width) * ratio))
        let nh = max(1, Int(Double(image.height) * ratio))

        // Resize into an RGBA buffer (row 0 = top of the image).
        var rgba = [UInt8](repeating: 0, count: nw * nh * 4)
        rgba.withUnsafeMutableBytes { buffer in
            let context = CGContext(
                data: buffer.baseAddress, width: nw, height: nh, bitsPerComponent: 8,
                bytesPerRow: nw * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            context.interpolationQuality = .medium
            context.draw(image, in: CGRect(x: 0, y: 0, width: nw, height: nh))
        }

        // 640 x 640 x 3, BGR, 0-255, padded with 114 (NHWC).
        var input = [Float](repeating: 114, count: size * size * 3)
        for y in 0..<nh {
            for x in 0..<nw {
                let p = (y * nw + x) * 4, q = (y * size + x) * 3
                input[q] = Float(rgba[p + 2])       // B
                input[q + 1] = Float(rgba[p + 1])   // G
                input[q + 2] = Float(rgba[p])       // R
            }
        }
        try interpreter.copy(input.withUnsafeBufferPointer { Data(buffer: $0) }, toInputAt: 0)
        try interpreter.invoke()
        let raw: [Float] = try interpreter.output(at: 0).data.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }

        var found: [Detection] = []
        var row = 0
        for stride in [8, 16, 32] {
            let cells = size / stride
            for gy in 0..<cells {
                for gx in 0..<cells {
                    let r = raw[(row * 7)..<(row * 7 + 7)].map { $0 }
                    row += 1
                    let cls = r[5] >= r[6] ? 0 : 1
                    let score = r[4] * r[5 + cls]
                    if score < thresholds[cls] { continue }
                    let s = Float(stride)
                    let cx = (r[0] + Float(gx)) * s, cy = (r[1] + Float(gy)) * s
                    let bw = exp(r[2]) * s, bh = exp(r[3]) * s
                    let k = Float(ratio)
                    found.append(Detection(
                        x1: min(max((cx - bw / 2) / k, 0), Float(image.width)),
                        y1: min(max((cy - bh / 2) / k, 0), Float(image.height)),
                        x2: min(max((cx + bw / 2) / k, 0), Float(image.width)),
                        y2: min(max((cy + bh / 2) / k, 0), Float(image.height)),
                        score: score, classId: cls))
                }
            }
        }
        return nms(found)
    }

    private func nms(_ boxes: [Detection], iouThreshold: Float = 0.45) -> [Detection] {
        var kept: [Detection] = []
        for cls in Set(boxes.map { $0.classId }).sorted() {
            // Stable sort: equal scores keep their original order.
            var order = boxes.enumerated().filter { $0.element.classId == cls }
                .sorted { $0.element.score != $1.element.score
                    ? $0.element.score > $1.element.score : $0.offset < $1.offset }
                .map { $0.element }
            while let best = order.first {
                kept.append(best)
                order = order.dropFirst().filter { iou(best, $0) <= iouThreshold }
            }
        }
        return kept.sorted { $0.score > $1.score }
    }

    private func iou(_ a: Detection, _ b: Detection) -> Float {
        let iw = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        let ih = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
        let inter = iw * ih
        let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return union > 0 ? inter / union : 0
    }
}
```

Run `detect` off the main thread; use one `Interpreter` per thread.

## Check your port

Each folder in `test_vectors/` has these files:

| File | Check |
|---|---|
| `input.png` | Feed this image to your code. |
| `letterboxed.png` | Your 640 × 640 canvas from step 2, stored as **RGB** for viewing (the model gets BGR). Yours should be within 1–2 grey levels. |
| `letterbox.json` | Should match your `ratio` and resized size. |
| `raw_output.npy` | The ONNX output for the reference canvas, `(1, 8400, 7)` float32. Use it to test steps 5–6 on their own. |
| `expected.json` | Final boxes at the recommended thresholds: `onnx` / `fp16` × `single` / `tiles2`. |

- **Pass criteria:** the same boxes and classes as `expected.json["fp16"]`, with coordinates
  within 2 px and scores within 0.05.
- **Your own resizer:** if it differs a little from the reference, compare with the
  tolerance rather than exactly.
- **`blank_grey`** must give no boxes.

Python check of the whole release (needs `onnxruntime`, `ai-edge-litert`, `numpy`,
`pillow`): `shasum -a 256 -c SHA256SUMS`, then run the quick start above on each
`input.png`.

## Good to know

- **Thresholds:** 0.10 for plates and 0.30 for vehicles are defaults. Raise the plate
  threshold if you see too many false boxes; lower it if you miss plates.
- **Single pass vs tiles:** skip step 7 if speed matters more than distant plates. It is
  about 5× faster and still finds most plates that are a reasonable size in the photo.
- **Known weak spots:** dark photos with the plate at a steep angle, tiny plates, and
  regions with few training plates. See `MODEL_CARD.md`.
- **Versions:** releases follow semver. A new MAJOR version changes the input, classes or
  output, so your code must change. MINOR means better weights with the same interface.
  PATCH means docs only.

## For maintainers

The training and release pipeline lives in this repository. The full spec is in
`docs/SPEC.md`; progress and decisions are in `docs/PROGRESS.md` and `docs/adr/`; data-pipeline
notes (sources, CCPD on Colab) are in `docs/development.md`.

```
make setup                  # Python 3.12 env with uv
make test && make lint
make try                    # run the release candidate on photos in data/my_photos/
make release                # build runs/<run>/release/v<version>/ and a zip
make release-check          # verify checksums and test vectors from a clean copy
```
