# When first use:
## 1. Install pip-pkg, config file, cmd-tool;

    cd xtele/xtele/scripts
    bash install_all.sh

## 2. Modify the config file.
Open the config file by cmd-lines

    xhumanoid-xtele --openconfig

# Calibrate the tele-operation table.
Using cali-tool and follow the terminal text prompt.

    xhumanoid-xtele --mode cali

# Auto select serial-name and add to config file.

    xhumanoid-xtele --mode systemtest

# Get current state

    xhumanoid-xtele --mode getstates

# Check the sign of the producted tele-arm. (Currently only supports Tienkung2)

    xhumanoid-xtele --mode checksign
