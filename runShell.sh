#!/bin/sh
#
#
#  Request 8G of RAM
#$   -l h_vmem=10G
#
#  This job executes matlab, only run if there are licenses available
#   -l matlab=1
#
#  Select cpus with model number 30 or higher
#$   -l cpumodel=30
#
#  The name shown in the qstat output and in the output file(s). The
#  default is to use the script name.
#  -N dont_make_this_too_long_as_only_the_first_few_characters_are_shown
#
#  The path used for the standard output stream of the job
#$ -o /is/ps/shared/users/asrikantha/std_out
#
# Merge stdout and stderr. The job will create only one output file which
# contains both the real output and the error messages.
#$ -j y
#
#  Use /bin/bash to execute this script
#$ -S /bin/bash
#
#  Run job from current working directory
#$ -cwd
#
#  Send email when the job begins, ends, aborts, or is suspended
#  -m 

exec ./HFTrainDetect $1 $2 $3
