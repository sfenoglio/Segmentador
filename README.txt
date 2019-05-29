%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
--------     CONTENTS     --------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

0- REQUIREMENTS
1- INSTALLATION AND DIRECTORIES
2- SETTINGS
3- DATASET
4- MODEL
5- OUTPUT FILES
6- EXAMPLES


=================================
0- REQUIREMENTS
=================================
- Python (Tested on Python 3.6)

- Python librarys:
  + argparse
  + json
  + numpy (1.15.2)
  + opencv-python (3.4.2.17)
  + opencv-contrib-python (3.4.2.17)
  + os
  + scikit-image (0.14.0)
  + scikit-learn (0.20.0)
  + torch (0.4.1)
  + torchvision (0.2.1)

** This tool was tested with these libraries versions. However, older versions
   could also work.


=================================
1- INSTALLATION AND DIRECTORIES
=================================
Installation only requires to unzip the "Segmentador.zip" file and download
the trained model with the link provided in modelo/link.txt (because it's larger
than maximum size allowed by github).
This tool can be launched by typing in the command line:

>> python3 segmentador.py --path_img <path_to_image_or_txt_file>

For example:

>> python3 segmentador.py --path_img images/example.txt
>> python3 segmentador.py --path_img images/98000117.11.tiff


Directories:
  config:            files to save and load default arguments.
  images:            images from Pki-3 dataset.
  model:             trained model.
  out:               folders with segmented images.
  src:               libraries and functions.


====================================
2- SETTINGS
====================================
Files with default arguments are stored in "config/".

--- main parameters ---

 path_img                   ---> path to the image file to segment or txt file
 path_out                   ---> path to the output folder
 path_arch                  ---> path to the file that defines model architecture
 path_model                 ---> path to the file that contains the trained parameters 
 device {cpu,gpu}           ---> run in cpu or gpu


All the parameters can be seen running the following command:

>> python3 segmentador.py -h

More details in "documentacion.pdf".
A typical configuration for all these parameters is provided with the software.


====================================
3- DATASET
====================================
Images files in "/images/" are from "Passau Chromosome Image Data, Pki-3", used in 
Gunter Ritter, Le Gao: Automatic segmentation of metaphase cells based on global 
context and variant analysis. Pattern Recognition 41 (2008) 38-55.

Link to the dataset: 
http://www.fim.uni-passau.de/en/faculty/former-professors/mathematical-stochastics/chromosome-image-data


=================================
4- MODEL
=================================
The model "modelo/entrenado" included was trained using the project EntrenamientoRed 
with synthetic data obtained from the project GenData.

Link to EntrenamientoRed project:
https://github.com/sfenoglio/EntrenamientoRed

Link to GenData project:
https://github.com/sfenoglio/GenData


=================================
5- OUTPUT FILES
=================================
Output files are placed in a new folder created in the 'path_out' with the name of the image file.
Inside it, the segmented chromosomes are saved in separated images with the name "<nn>_<cc>.png":
 <nn>    cluster number detected
 <cc>    class number 


=================================
6- EXAMPLES
=================================
In 'path_out' there are some outputs of segmented images listed in "images/example.txt".
The command to repeat the process is simply:

>> python3 segmentador.py

