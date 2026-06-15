import os

print("Rebuilding MARL Traffic Environment (Smart TLS)...")

# 1. The Geometry Fix
# We replaced '-j traffic_light' with '--tls.guess'
# This ensures only the central 4-way intersections become AI agents.
os.system("netgenerate --grid --grid.number 2 --grid.length 200 --grid.attach-length 100 --tls.guess --default.lanenumber 2 --default.speed 13.89 -o my_grid.net.xml")

# 2. Locate the SUMO randomTrips tool
sumo_home = os.environ.get("SUMO_HOME")
random_trips = f'"{sumo_home}\\tools\\randomTrips.py"'

# 3. Regenerate the routes for the new smart grid
os.system(f"python {random_trips} -n my_grid.net.xml -o my_routes.rou.xml --end 3600 --period 2.0 --fringe-factor 10 --validate")

print("\nNetwork fixed! All agents are now homogeneous.")