#!/usr/bin/env bash

# rsync -avu --delete -e 'ssh -p 10022' /data1/BigCodeModel/LMFlow cse30001192@172.18.34.19:~/zzr/ 
rsync -avu -e 'ssh -p 10022' /data1/BigCodeModel/LMFlow cse30001192@172.18.34.19:~/zzr/ 
rsync -avu -e 'ssh -p 1022' /data1/BigCodeModel/LMFlow cg@172.18.36.130:/data/cg 

