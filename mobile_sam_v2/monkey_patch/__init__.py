import importlib.util

def dynamic_patch_ultralytics():
    ultralytics_installed = importlib.util.find_spec("ultralytics") is not None
    if ultralytics_installed:
        from .ultralytics import patch_ultralytics
        print("mobile_sam_v2: patching ultralytics")
        patch_ultralytics()
        print("mobile_sam_v2: ultralytics patched")
    else:
        print("mobile_sam_v2: skipping ultralytics patch as ultralytics is not installed")

def patch_all():
    dynamic_patch_ultralytics()
