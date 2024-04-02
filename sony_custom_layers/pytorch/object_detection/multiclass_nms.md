<h2>sony_custom_layers.pytorch.multiclass_nms</h2>

[source](nms.py)
<p>Multi-class non-maximum suppression.<br>
Detections are returned in descending order of their scores.
The output tensors always contain a fixed number of detections, as defined by 'max_detections'.
If fewer detections are selected, the output tensors are zero-padded up to 'max_detections'.</p>

```
from sony_custom_layers.pytorch import multiclass_nms

multiclass_nms(boxes: Tensor, scores: Tensor, score_threshold: float, iou_threshold: float, max_detections: int)
```
<p><strong>Arguments:</strong></p>
<ul><strong>boxes</strong> (Tensor) Input boxes with shape [batch, n_boxes, 4], specified in corner coordinates (x_min, y_min, x_max, y_max). 
Agnostic to the x-y axes order.</ul>
<ul><strong>scores</strong> (Tensor) Input scores with shape [batch, n_boxes, n_classes].</ul>
<ul><strong>score_threshold</strong> (float) The score threshold. Candidates with scores below the threshold are discarded.</ul>
<ul><strong>iou_threshold</strong> (float) The Intersection Over Union (IOU) threshold for boxes overlap.</ul>
<ul><strong>max_detections</strong> (int) The number of detections to return.</ul>
<p><strong>Returns:</strong></p>
<ul>
    'NMSResults' named tuple:
        <ul><strong>boxes</strong> (Tensor): The selected boxes with shape [batch, max_detections, 4]</ul>
        <ul><strong>scores</strong> (Tensor): The corresponding scores in descending order with shape [batch, max_detections]</ul>
        <ul><strong>labels</strong> (Tensor): The labels for each box with shape [batch, max_detections]</ul>
        <ul><strong>n_valid</strong> (Tensor): The number of valid detections out of 'max_detections' with shape [batch, 1]</ul>
</ul>
<p><strong>Raises:</strong></p>
<ul>ValueError: Invalid arguments are passed or input tensors with unexpected shape are received.</ul>

<p><strong>NMSResults</strong> also provides the following methods: 
<ul><strong>detach()</strong> - detach all tensors and return a new NMSResults object</ul> 
<ul><strong>cpu()</strong> - move all tensors to cpu and return a new NMSResults object</ul> 
<ul><strong>apply(f: Callable[[Tensor], Tensor])</strong> - apply a function f to all tensors and return a new NMSResults object</ul>
