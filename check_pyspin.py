import PySpin

system = PySpin.System.GetInstance()

version = system.GetLibraryVersion()
print(
    f"Spinnaker version: "
    f"{version.major}.{version.minor}.{version.type}.{version.build}"
)

cams = system.GetCameras()
print("Number of cameras detected:", cams.GetSize())

cams.Clear()
system.ReleaseInstance()
