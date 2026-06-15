import os

print("Rebuilding MARL Traffic Environment...")

# 1. Regenerate the grid WITH fringe roads
# The attach-length flag adds outer roads, forcing the corners to have cross-traffic.
print("Generating geometry...")
os.system("netgenerate --grid --grid.number 2 --grid.length 200 --grid.attach-length 100 --default.lanenumber 2 --default.speed 13.89 -j traffic_light -o my_grid.net.xml")

# 2. Locate the SUMO randomTrips tool securely
sumo_home = os.environ.get("SUMO_HOME")
random_trips = f'"{sumo_home}\\tools\\randomTrips.py"'

# 3. Regenerate the routes to use the new outer edges
print("Generating stochastic traffic...")
os.system(f"python {random_trips} -n my_grid.net.xml -o my_routes.rou.xml --end 3600 --period 2.0 --fringe-factor 10 --validate")

print("\nEnvironment rebuilt successfully! Ready for AI extraction.")