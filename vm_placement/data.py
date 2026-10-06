"""Problem data for Question 9: 5 physical machines (PMs) and 5 virtual machines (VMs).

Edit the two tables below to match the values on your question sheet. Everything
else in the project (fitness, algorithms, report) reads its numbers from here.
"""

# name: (cpu cores, ram GB, bandwidth Gbps, power W)
PMS = {
    "PM1": (16, 32, 10, 400),
    "PM2": (32, 64, 20, 600),
    "PM3": (24, 48, 10, 500),
    "PM4": (8, 16, 5, 250),
    "PM5": (16, 64, 10, 450),
}

# name: (cpu cores, ram GB, bandwidth Gbps)
VMS = {
    "VM1": (4, 8, 1),
    "VM2": (8, 12, 2),
    "VM3": (8, 16, 2),
    "VM4": (2, 4, 1),
    "VM5": (12, 16, 3),
}

RESOURCES = ("CPU", "RAM", "BW")
UNITS = ("cores", "GB", "Gbps")
