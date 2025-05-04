from ultralytics import SAM

model = SAM("sam2.1_t.pt")

model.info()

results = model("image_user/155347 (1).jpg")
