"""Rebuild sumo/intersection.net.xml (only needed if you edit the node/edge files)."""
import os, shutil, subprocess, sys
exe = shutil.which("netconvert")
if exe is None:
    import sumo
    exe = os.path.join(os.path.dirname(sumo.__file__), "bin", "netconvert")
subprocess.check_call([exe, "--node-files=sumo/nodes.nod.xml", "--edge-files=sumo/edges.edg.xml",
                       "--output-file=sumo/intersection.net.xml"])
print("Network rebuilt.")
