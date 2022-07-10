Body-Size-Estimator
==============================

The Model Take the Image and Height of the Person and calculates the Shoulder-size,chest-size,sleeve-length,cloth-length(Upper),neck-size,waist-size

Project Organization:Simuverse ML-Team(Devanshu Mishra)
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    │
    ├── models             <- Should Contains Pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt,
    │                         pose_iter_160000.caffemodel(Take It From Google Drive)
    │
    │
    ├── references         <- Contains sh get_model.sh caffemodel url
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Contains the Sample Image and Result_Sample Image used in model Testing
    │   │   
    │   │
    │   │
    │   ├── models         <- Contains Scripts Pose_Estimation_model and Flask api
    │   │   │                
    │   │   ├── run_pose.py
    │   │   └── app.py
    │   │ 

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Code for running OpenPose code for human pose estimation using deep learning in OpenCV. 

First step is download the pretrained models:

# On Ubuntu/Mac/Win, run this command to download the models.
sh get_model.sh

or Drive Link:https://drive.google.com/drive/folders/1rhez5b0cFtK3EBA-jn1ekNeJ93JlOXeE?usp=sharing

Now, we can just run these models using the commands below. There are three pretrained models available each trained on a different dataset. In the blog, we have only shown MPI but you can run the other two as well. 

## To Run The Model(run_pose.py):

To Run MPI pretrained model on an image sample.jpg:

```
python run_pose.py --input sample.jpg --proto pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model pose/mpi/pose_iter_160000.caffemodel --dataset MPI
```

The output results are shown in a new image result_sample.jpg saved on your hard disk.

## To Run Flask API(app.py)

```
python app.py
```

