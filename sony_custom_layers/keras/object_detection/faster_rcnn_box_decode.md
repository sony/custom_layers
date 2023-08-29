<h2>sony_custom_layers.keras.object_detection.FasterRCNNBoxDecode</h2>

[source](faster_rcnn_box_decode.py)
<p>Box decoding as per <a href="https://arxiv.org/abs/1506.01497">Faster R-CNN</a>.</p>
<p>Example:</p>

```
from sony_custom_layers.keras.object_detection import FasterRCNNBoxDecode

box_decode = FasterRCNNBoxDecode(anchors, scale_factors=(10, 10, 5, 5), clip_window=(0, 0, 1, 1))

decoded_boxes = box_decode(rel_codes)
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
        <td>Scaling factors in the format (y, x, height, width). Type: floats or integers.</td>
    </tr>
    <tr>
        <td>clip_window</td>
        <td>Clipping window in the format (y_min, x_min, y_max, x_max). Type: floats or integers.</td>
    </tr>
</table>

<p><strong>Inputs:</strong></p>
<ul>
    <li>Relative codes (encoded offsets) with a shape of (batch, n_boxes, 4) in centroid coordinates (y_center, x_center, h, w).</li>
</ul>
<p><strong>Returns:</strong></p>
<ul>
    <li>Decoded boxes with a shape of (batch, n_boxes, 4) in corner coordinates (y_min, x_min, y_max, x_max).</li>
</ul>
<p><strong>Raises:</strong></p>
<ul>
    <li>ValueError: If an input tensor with an unexpected shape is received.</li>
</ul>

</body>
</html>
