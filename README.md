# MDImageNet-CE Script Note
- 2026-06-20 outline

***
## Processing
### [Porcessing_1 : Polygon to bounding box and filename hashing](./script/Processing_1_Parse_Check_org-xml-polygon-points.py)
### [Porcessing_2 : SAM-Derived Bounding Box Harmonization](./script/Processing_2_Refine-BBox_pre_10K-to-refinement_10K-xml_twcc.py)
### 


***
## Dataset format converting

### [Porcessing_3 : Harmonized BBox to PSACAL VOC](./script/Processing_3_BBox2VOC.py)
### VOC to COCO
### COCO to YOLO



***
## Visualization
### GT and SAMBbox ploting
### MD classes (Exp/Con)
###



***
## Model Training
### MS COCO with RF-DETR
#### RF-DETRs
#### RF-DETRm
#### map log

### YOLO with YOLO
#### YOLOv11n
#### YOLO26n
#### YOLO26x
#### map log



*** 
## Check
### some check scripts...
#### Check BBox of single image from original pre_10K.xml.
Script[Check.BBox](./script/Check_Draw_xml_4points_annotations_for-TLBR2points.py)

<img src="./img/Check_Draw_xml_4points_annotations.jpg" width="300">

