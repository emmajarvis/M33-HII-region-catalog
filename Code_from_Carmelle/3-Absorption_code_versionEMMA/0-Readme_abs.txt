


Ppxf will do an absorption galaxy pop. model with the 
integrated spectrum (in specific galaxy structures as desired)
of selected pixels (based on a low Ha emission and good SNR_cont3, as desired) 
at zero velocity (so you need a velocity map covering the selected pixels).


1-Velocity.ipynb

Create a velocity map for all pixels (including those not well measured
with orcs) using the SN3 velocity map and known parameters for the galaxy rotation.


2-Galaxy_structures.ipynb

Split the galaxy in different structures...
Either to link emission regions to a structures or to define various galaxy stellar pop...


3-Absorption.ipynb

First install ppxf :  https://pypi.org/project/ppxf/8.2.6/  
For the moment we use version 8.2.6, but a new one exists

In a terminal:
conda activate orb3 ''or your environment orb''

pip install ppxf==8.2.6


In the ppxf repertory, exchange the ppxf_util.py with the new version that takes into account the
SITELLE sync
Ask CR for the new version.
The path to ppxf is usually: miniconda3/envs/orb3/lib/python3.10/site-packages/ppxf


Inside this repertory: create the repertory SSP 
Extract a SSP library from MILES UN_baseFE  at
https://cloud.iac.es/index.php/s/aYECNyEQfqgYwt4?path=%2FMILES
And copy it into SSP.
This library could be improved to match the SNR R (with INDO...) and the latest version of ppxf...


In 3-Absorption.ipynb
Ajuste the path to the library

