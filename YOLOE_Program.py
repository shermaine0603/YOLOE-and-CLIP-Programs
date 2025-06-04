from ultralytics import YOLOE

# Prompt free (YOLOE model)
model = YOLOE("yoloe-11l-seg-pf.pt")

results = model.predict(
    "Photo 5.jpg",
    show=True,
    save=True,
    show_conf=False
)

list = []
# Extract the results
for r in results:
    boxes = r.boxes.xyxy.cpu().tolist()
    cls = r.boxes.cls.cpu().tolist()
    
    # Display results
    for b, c in zip(boxes, cls):
        print(f"box: {b}, cls: {model.names[int(c)]}")
        list.append(model.names[int(c)])

# for i in list:
#     print("There is a", i)

print(list)

string = ', '.join(list)
print(string)