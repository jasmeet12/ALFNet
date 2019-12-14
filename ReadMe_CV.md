**Read ME For ALFNet and CSP code**
Markup: *For CrowdHuman dataset file https://github.com/jasmeet12/ALFNet/blob/master/resize_annotation.py is used to resize the images and translate the ground box to resized images. Only for ALFNet once the dataset is generated we will copy it to CSP.

  *Please make sure crowdData validation and annotation is copied to crowdHuman/val/
      *it will output resized images to crowdHuman/val/images

*generate_data_crowd.py will generate the pickle file for the crowdHuman Dataset.

*plotBoxes and plotCrowdBoxes file is used to add groundTruth and Predicted boxes to the images. It will result output to /output/images_dt and /output/images_gt for predicted and ground truth respectivily 

*create_gt.json.py script is used to change annotation.odgt(ground truth annotation) file to json file in the same format of cityperson annotation ground truth json file

  *it output val_crowd_gt.json file inside evaluation folder.
  
