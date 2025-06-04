from ultralytics import YOLOE

# Prompt free (YOLOE model)
model = YOLOE("yoloe-11l-seg-pf.pt")
results = model(0,show=True)

for result in results:
    boxes = result.boxes
    classes = result.names
