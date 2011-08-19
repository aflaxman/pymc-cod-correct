
import data
import models
import os 
import re
import subprocess

outdir = '/home/j/Project/Causes of Death/Under Five Deaths/Cod Correct Output'
indir = '/home/j/Project/Causes of Death/Under Five Deaths/CoD Correct Input Data' 

folders = [folder for folder in os.listdir(indir) if re.search('v02', folder)]
countries = [s[9:12] for s in folders]
ages = ['Early_Neonatal', 'Late_Neonatal', 'Post_Neonatal', '1_4', 'Under_Five']

countries = countries[:3]

for age in ages: 
    for iso3 in countries: 
        for sex in ['M', 'F']: 
            jobname = 'cc%s_%s_%s' % (iso3, sex, age)
            call = 'qsub -cwd -N %s cluster_shell.sh cluster_fit.py "%s" "%s" "%s"' % (jobname, age, iso3, sex)
            subprocess.call(call, shell=True)  

