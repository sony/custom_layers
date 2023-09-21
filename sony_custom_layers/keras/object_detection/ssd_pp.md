<h2>sony_custom_layers.keras.object_detection.SSDPostProcess</h2>

[source](ssd_post_process.py)
<p>SSD Post Processing, based on <a href="https://arxiv.org/abs/1512.02325">https://arxiv.org/abs/1512.02325</a>.</p>
<p>Example:</p>

```
from sony_custom_layers.keras.object_detection import SSDPostProcessing, ScoreConverter

post_process = SSDPostProcess(anchors=anchors,
                              scale_factors=(10, 10, 5, 5),
                              clip_size=(320, 320),
                              score_converter=ScoreConverter.SIGMOID,
                              score_threshold=0.01,
                              iou_threshold=0.6,
                              max_detections=200,
                              remove_background=True)

boxes, scores, labels, n_valid = post_process([encoded_offsets, logits])
```

<table>
    <tr>
        <th>Argument</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>anchors</td>
        <td>Anchors with a shape of (n_boxes, 4) in corner coordinates (y_min, x_min, y_max, x_max).</td>
    </tr>
    <tr>
        <td>scale_factors</td>
        <td>Box decoding scaling factors in the format (y, x, height, width). Type: floats or integers.</td>
    </tr>
    <tr>
        <td>clip_size</td>
        <td>Clipping size in the format (height, width). The decoded boxes are clipped to the range y=[0, height] and x=[0, width]. Typically, the clipping size is (1, 1) for normalized boxes and the image size for boxes in pixel coordinates. Type: floats or integers.</td>
    </tr>
    <tr>
        <td>score_converter</td>
        <td>Conversion to apply to the input logits (sigmoid, softmax, or linear). Type: sony_custom_layers.keras.object_detection.ScoreConverter or string. </td>
    </tr>
    <tr>
        <td>score_threshold</td>
        <td>Score threshold for non-maximum suppression. Type: float.</td>
    </tr>
    <tr>
        <td>iou_threshold</td>
        <td>Intersection over union threshold for non-maximum suppression. Type: float.</td>
    </tr>
    <tr>
        <td>max_detections</td>
        <td>The number of detections to return. Type: integer.</td>
    </tr>
    <tr>
        <td>remove_background</td>
        <td>If True, the first class is removed from the input scores (after the score_converter is applied). Type: boolean.</td>
    </tr>
</table>

<p><strong>Inputs:</strong></p>
<ul>
    <li>A list or tuple consisting of (rel_codes, scores).
        <ul>
            <li>0: Relative codes (encoded offsets) with a shape of (batch, n_boxes, 4) in centroid coordinates (y_center, x_center, w, h).</li>
            <li>1: Scores or logits with a shape of (batch, n_boxes, n_labels).</li>
        </ul>
    </li>
</ul>
<p><strong>Returns:</strong></p>
<ul>
    <li>0: Selected boxes sorted by scores in descending order, with a shape of (batch, max_detections, 4), in corner coordinates (y_min, x_min, y_max, x_max).</li>
    <li>1: Scores corresponding to the selected boxes, with a shape of (batch, max_detections).</li>
    <li>2: Labels corresponding to the selected boxes, with a shape of (batch, max_detections). Each label corresponds to the class index of the selected score in the input scores.</li>
    <li>3: The number of valid detections out of max_detections.</li>
</ul>
<p><strong>Raises:</strong></p>
<ul>
    <li>ValueError: If provided input tensors have unexpected or non-matching shapes.</li>
</ul>

</body>
</html>

