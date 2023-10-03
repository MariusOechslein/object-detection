# Object Detection algorithms from scratch with Pytorch.

The goal of this repository was to understand the concepts of objects detection with Pytorch more deeply by implementing everything from scratch. 

### 0. Metrics

- Intesection over union
- Mean average precision
- Non max suppression

  

### 1. YOLOv1 paper

```
@inproceedings{redmon2016you,
  title={You only look once: Unified, real-time object detection},
  author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={779--788},
  year={2016}
}
```

Link: [https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)

- YOLOv1 model architecture: ![image](https://github.com/MariusOechslein/object-detection/assets/67323507/72b0dfea-943c-4b09-bb67-abbf673edde6)
- YOLOv1 Multi-part loss function: ![image](https://github.com/MariusOechslein/object-detection/assets/67323507/9b36cb77-d2c3-4833-9921-85dbe35037c9)


### Sample output

Sample output after only **10** training epochs:

![image](https://github.com/MariusOechslein/object-detection/assets/67323507/fc88c040-3096-4ccc-8a51-c9bfa7d3a240)


