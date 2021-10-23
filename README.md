# point2color
Point2color: 3D Point Cloud Colorization Using a Conditional Generative Network and Differentiable Rendering for Airborne LiDAR [Earth Vision 2021](http://www.classic.grss-ieee.org/earthvision2021/program.html)

[Papaer]( https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Shinohara_Point2color_3D_Point_Cloud_Colorization_Using_a_Conditional_Generative_Network_CVPRW_2021_paper.html)  
[Project]( https://shnhrtkyk.github.io/portfolio/portfolio-5/)

![Point2color](https://user-images.githubusercontent.com/49538481/118940135-5efa8080-b98b-11eb-8c22-de4f156c3fa7.png)

## Environment
The following libraries are the environment I used for my experiments.

- python 3.6.5
- cuda: 10.1 
- nccl: 2.2.13
- cudnn: 7.4
- librarys
    - laspy==1.7.0
    - torch==1.6.0+cu101
    - torch-cluster==1.5.8
    - torch-geometric==1.6.3
    - torch-scatter==2.0.5
    - torch-sparse==0.6.8
    - torch-spline-conv==1.2.0
    - torchfile==0.1.0
    - torchvision==0.7.0+cu101
    - pytorch3d==0.3.0
    - open3d==0.8.0.0
    

@InProceedings{Shinohara_2021_CVPR,  
    author    = {Shinohara, Takayuki and Xiu, Haoyi and Matsuoka, Masashi},  
    title     = {Point2color: 3D Point Cloud Colorization Using a Conditional Generative Network and Differentiable Rendering for Airborne LiDAR},  
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},  
    month     = {June},  
    year      = {2021},  
    pages     = {1062-1071}  
}