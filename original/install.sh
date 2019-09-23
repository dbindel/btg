#!/bin/sh

# Script to configure installation options for btg program

echo 'Enter path to install btg and tkbtg: '
read btg_location

echo 'Enter location for the btg help file: '
read btghelp_location

echo "#!/bin/sh

# Tcl/Tk front-end to the btg program
# Written by David Bindel

# This is a trick to get sh to start wish regardless of where
# it is located from system to system.
# \\
exec wish -f \"\$0\" \${1+\"\$@\"}

##############################################################
# System specific settings
##############################################################

# These settings should be tailored for your installation
# Set the btg_program variable to the fully pathed name of the
#   btg engine program
# Set the btg_help variable to the fully pathed name of the
#   btg help file

set btg_program \"$btg_location/btg\"
set btg_help \"$btghelp_location/btghelp.txt\"

" > tkbtg

cat tkbtg.main >> tkbtg

chmod +x tkbtg

/bin/csh copy.csh $btg_location $btghelp_location
